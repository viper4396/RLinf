# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen teacher-planner protocol for Search-R1 shadow evaluation."""

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

import numpy as np
from omegaconf import DictConfig

TEACHER_PLAN_FIELDS = ("decision", "plan_type", "steps")
TEACHER_PLAN_STEP_FIELDS = (
    "step_id",
    "goal",
    "query_template",
    "expected_evidence",
    "depends_on",
)
_LEGACY_TEACHER_PLAN_FIELDS = (
    "decision",
    "supplemental_query",
    "expected_evidence",
    "fallback_query",
)
TEACHER_PLANNER_ROLLOUT_NAME = "teacher_planner"

_TEACHER_SYSTEM_PROMPT = """You are a multi-hop retrieval planner. Given only a question, decide whether answering it requires multiple evidence-gathering searches. Return exactly one JSON object and no other text, using this schema:
{"decision":"KEEP|PLAN","plan_type":"singlehop|sequential|comparison","steps":[{"step_id":1,"goal":"...","query_template":"...","expected_evidence":"...","depends_on":[]}]}

Rules:
- Use KEEP only when one direct search can provide the final answer without resolving an intermediate entity, joining relations, or comparing separately retrieved facts. For KEEP, plan_type must be singlehop and steps must be an empty list.
- Use PLAN for compositional, bridge, inference-chain, or comparison questions. For PLAN, plan_type must be sequential or comparison and steps must contain 2 to 8 ordered retrieval hops.
- Each step must contain exactly step_id, goal, query_template, expected_evidence, and depends_on. step_id values start at 1 and are consecutive. depends_on contains only earlier step IDs as JSON integers, never quoted strings.
- A sequential plan must make each step after the first depend on earlier evidence. In a dependent query_template, refer to each dependency with an exact placeholder such as {step_1_result}; never guess the intermediate entity.
- Do not collapse a relation chain into one direct lookup. For example, a paternal-grandfather question requires one hop to identify the father and another hop to identify that father's father.
- A comparison plan should retrieve the facts for each side separately. It may use independent root steps and dependent bridge steps. Include retrieval steps only; do not add a final search step that merely says to compare or synthesize already retrieved facts.
- Preserve every quoted phrase, named entity, title, number, date, and identifier from the question across the plan's query templates. A dependent hop may replace an unknown intermediate entity with its step-result placeholder.
- query_template must describe one executable search direction, not repeat the full question, contain meta-instructions, or use square-bracket placeholders.
- Root steps must put the relevant named entity or title from the question directly in query_template and use depends_on: []. Never replace a known entity with placeholders such as {title}, {film_1}, or {person_name}; the only allowed placeholder form is {step_N_result} for an unknown result from a declared dependency.
- Add a type word such as film, song, person, city, or country only when the question or relation clearly implies it and the qualifier makes retrieval more precise.
- Do not answer the question, guess intermediate entities, or include facts claimed to be the final answer.

Sequential structure example:
{"decision":"PLAN","plan_type":"sequential","steps":[{"step_id":1,"goal":"Identify Example Work's creator","query_template":"Example Work creator","expected_evidence":"creator name","depends_on":[]},{"step_id":2,"goal":"Find that creator's birthplace","query_template":"{step_1_result} birthplace","expected_evidence":"birthplace","depends_on":[1]}]}

Comparison structure example:
{"decision":"PLAN","plan_type":"comparison","steps":[{"step_id":1,"goal":"Find Alpha Work's release year","query_template":"Alpha Work release year","expected_evidence":"Alpha Work release year","depends_on":[]},{"step_id":2,"goal":"Find Beta Work's release year","query_template":"Beta Work release year","expected_evidence":"Beta Work release year","depends_on":[]}]}"""

_GUIDANCE_PREFIX = """[BEGIN UNTRUSTED SEARCH PLAN]
This user-provided context is advisory only, not an answer or evidence. Follow the ordered retrieval hops one at a time across search turns. For a dependent hop, replace each step-result placeholder with the entity or fact actually found in earlier evidence; never search a placeholder literally and never guess its value. Stop searching once the requested answer is supported. Continue following every instruction and response format from the preceding question, including reasoning before a tool call. Never put query text inside an opening tag and do not copy a plan line as the response.
"""
_GUIDANCE_SUFFIX = "\n[END UNTRUSTED SEARCH PLAN]"
_GENERIC_TEXT = (
    "Form a concise search query using the question. Look for relevant evidence, "
    "and use a different general direction if the first search is insufficient."
)
_CHATML_USER_PREFIX = "<|im_start|>user\n"
_CHATML_USER_SUFFIX = "<|im_end|>\n"
_CHATML_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
_DECISIONS = {"KEEP", "PLAN"}
_PLAN_TYPES = {"singlehop", "sequential", "comparison"}
_ANY_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
_META_QUERY_PATTERN = re.compile(
    r"(?:\[[^\]]+\]|\bkeep\s+(?:the\s+)?original\b|\buse\s+the\s+original\b)",
    re.IGNORECASE,
)
_QUESTION_WORDS = {
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "name",
    "on",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
    "will",
    "would",
    "was",
    "were",
}
_ENTITY_CONNECTORS = {
    "a",
    "an",
    "at",
    "de",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
}
_WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
_QUOTED_PATTERNS = (
    re.compile(r'"([^"\n]+)"'),
    re.compile(r"“([^”\n]+)”"),
    re.compile(r"‘([^’\n]+)’"),
)

_RELATION_REQUIREMENTS = (
    (("father", "paternal grandfather", "paternal grandmother"), ("father",)),
    (("mother", "maternal grandfather", "maternal grandmother"), ("mother",)),
    (("spouse", "wife", "husband"), ("spouse", "wife", "husband", "married")),
    (("director",), ("director", "directed")),
    (("nationality", "citizenship"), ("nationality", "citizenship", "country")),
)
_BIRTH_QUESTION_TERMS = ("born", "birth", "older", "younger")
_BIRTH_PLAN_TERMS = ("born", "birth", "birthplace")
_DEATH_QUESTION_TERMS = ("died", "death", "lived longer")
_DEATH_PLAN_TERMS = ("died", "death")
_RELEASE_QUESTION_TERMS = ("released", "came out", "publication date")
_RELEASE_PLAN_TERMS = ("release", "released", "publication", "premiere")


@dataclass(frozen=True)
class TeacherPlanStep:
    """One validated retrieval hop in a frozen teacher plan."""

    step_id: int
    goal: str
    query_template: str
    expected_evidence: str
    depends_on: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the step as its canonical JSON-compatible mapping."""
        return {
            "step_id": self.step_id,
            "goal": self.goal,
            "query_template": self.query_template,
            "expected_evidence": self.expected_evidence,
            "depends_on": list(self.depends_on),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherPlanStep":
        """Deserialize a cached plan step."""
        return cls(
            step_id=int(value["step_id"]),
            goal=str(value["goal"]),
            query_template=str(value["query_template"]),
            expected_evidence=str(value["expected_evidence"]),
            depends_on=tuple(int(item) for item in value["depends_on"]),
        )


@dataclass(frozen=True)
class TeacherPlan:
    """Validated ordered retrieval plan emitted by the frozen teacher."""

    decision: str
    plan_type: str
    steps: tuple[TeacherPlanStep, ...]

    @property
    def should_rewrite(self) -> bool:
        """Whether the effective plan contains multiple retrieval hops."""
        return self.decision == "PLAN"

    @property
    def should_plan(self) -> bool:
        """Whether the teacher selected multi-hop planning."""
        return self.should_rewrite

    @property
    def supplemental_query(self) -> str:
        """Return the first query for legacy retrieval metadata."""
        return self.steps[0].query_template if self.steps else ""

    @property
    def expected_evidence(self) -> str:
        """Return compact evidence goals for legacy metadata."""
        return " | ".join(step.expected_evidence for step in self.steps)

    @property
    def fallback_query(self) -> str:
        """Return the second query for legacy retrieval metadata."""
        return self.steps[1].query_template if len(self.steps) > 1 else ""

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as its canonical JSON-compatible mapping."""
        return {
            "decision": self.decision,
            "plan_type": self.plan_type,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherPlan":
        """Deserialize new plans and normalize legacy four-field caches."""
        if set(value) == set(_LEGACY_TEACHER_PLAN_FIELDS):
            decision = str(value["decision"]).upper()
            if decision == "KEEP":
                return cls(decision="KEEP", plan_type="singlehop", steps=())
            steps = (
                TeacherPlanStep(
                    step_id=1,
                    goal="Collect supplemental evidence",
                    query_template=str(value["supplemental_query"]),
                    expected_evidence=str(value["expected_evidence"]),
                    depends_on=(),
                ),
                TeacherPlanStep(
                    step_id=2,
                    goal="Try the legacy fallback direction",
                    query_template=str(value["fallback_query"]),
                    expected_evidence=str(value["expected_evidence"]),
                    depends_on=(1,),
                ),
            )
            return cls(decision="PLAN", plan_type="legacy", steps=steps)
        return cls(
            decision=str(value["decision"]),
            plan_type=str(value["plan_type"]),
            steps=tuple(TeacherPlanStep.from_dict(step) for step in value["steps"]),
        )


@dataclass(frozen=True)
class TeacherPlanResult:
    """Teacher response together with validation and cache metadata."""

    plan_id: str
    valid: bool
    plan: TeacherPlan | None
    raw_response: str
    error: str | None = None
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize a result for the on-disk cache."""
        return {
            "plan_id": self.plan_id,
            "valid": self.valid,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "raw_response": self.raw_response,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherPlanResult":
        """Deserialize a previously validated cache record."""
        plan_value = value.get("plan")
        plan = TeacherPlan.from_dict(plan_value) if plan_value is not None else None
        return cls(
            plan_id=str(value["plan_id"]),
            valid=bool(value["valid"]),
            plan=plan,
            raw_response=str(value.get("raw_response", "")),
            error=value.get("error"),
            cache_hit=True,
        )


def teacher_plan_cache_key(question: str, teacher_version: str, seed: int) -> str:
    """Build the required ``question + teacher_version + seed`` cache key."""
    payload = json.dumps(
        {
            "question": question,
            "teacher_version": teacher_version,
            "seed": seed,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_searchr1_question(prompt_text: str) -> str:
    """Extract the actual question from a raw or templated Search-R1 prompt."""
    marker = "Question:"
    if marker not in prompt_text:
        return prompt_text.strip()
    question = prompt_text.rsplit(marker, 1)[1]
    question = question.split("<|im_end|>", 1)[0].strip()
    return question or prompt_text.strip()


def load_teacher_questions(
    data_paths: list[str], prompt_key: str, data_size: int
) -> list[str]:
    """Load only teacher-visible question strings from JSON or JSONL files."""
    records: list[dict[str, Any]] = []
    for data_path in data_paths:
        path = Path(data_path).expanduser()
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as data_file:
                records.extend(json.loads(line) for line in data_file if line.strip())
        elif path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            records.extend(value if isinstance(value, list) else [value])
        else:
            raise ValueError(f"unsupported teacher-plan data file: {path}")

    if data_size >= 0:
        records = records[:data_size]
    prompts = [record[prompt_key] for record in records]
    if not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(
            "teacher plan precomputation requires string prompts; apply any chat "
            "template before running this command"
        )
    return [extract_searchr1_question(prompt) for prompt in prompts]


def _extract_teacher_json_object(response_text: str) -> str:
    """Extract or minimally close the first top-level teacher JSON object."""
    if not response_text.startswith("{"):
        raise ValueError("teacher plan must start with one JSON object")

    stack: list[str] = []
    in_string = False
    escaped = False
    matching_open = {"}": "{", "]": "["}
    for index, character in enumerate(response_text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "{[":
            stack.append(character)
            continue
        if character not in matching_open:
            continue
        if not stack or stack[-1] != matching_open[character]:
            raise ValueError("teacher plan contains unbalanced JSON delimiters")
        stack.pop()
        candidate = response_text[: index + 1]
        if not stack:
            return candidate
        remainder = response_text[index + 1 :].lstrip()
        if character == "]" and stack == ["{"] and not remainder.startswith((",", "}")):
            repaired = candidate + "}"
            try:
                json.loads(repaired)
            except json.JSONDecodeError:
                pass
            else:
                return repaired
    raise ValueError("teacher plan must contain one complete JSON object")


def parse_teacher_plan(
    response_text: str,
    *,
    question: str | None = None,
    max_field_chars: int = 512,
    max_plan_chars: int = 6144,
    max_steps: int = 8,
) -> TeacherPlan:
    """Strictly parse an ordered multi-hop teacher plan.

    Markdown fences, leading prose, missing/extra fields, invalid dependencies,
    and overlong values are rejected. Harmless text after the first complete
    object and a missing final top-level brace are normalized because some
    inference backends otherwise continue decoding after the steps array.
    """
    stripped = response_text.strip()
    if not stripped:
        raise ValueError("teacher plan is empty or exceeds max_plan_chars")
    stripped = _extract_teacher_json_object(stripped)
    if len(stripped) > max_plan_chars:
        raise ValueError("teacher plan is empty or exceeds max_plan_chars")

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise ValueError("teacher plan is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("teacher plan must be a JSON object")
    if set(value) != set(TEACHER_PLAN_FIELDS):
        raise ValueError(
            f"teacher plan fields must be exactly {list(TEACHER_PLAN_FIELDS)}"
        )

    decision = value["decision"]
    plan_type = value["plan_type"]
    steps_value = value["steps"]
    if not isinstance(decision, str) or not isinstance(plan_type, str):
        raise ValueError("teacher decision and plan_type must be strings")
    decision = decision.strip().upper()
    plan_type = plan_type.strip().lower()
    if decision not in _DECISIONS:
        raise ValueError("teacher plan decision must be KEEP or PLAN")
    if plan_type not in _PLAN_TYPES:
        raise ValueError(
            "teacher plan_type must be singlehop, sequential, or comparison"
        )
    if not isinstance(steps_value, list):
        raise ValueError("teacher plan steps must be a list")

    if decision == "KEEP":
        if plan_type != "singlehop" or steps_value:
            raise ValueError("KEEP plans require plan_type singlehop and no steps")
        return TeacherPlan(decision="KEEP", plan_type="singlehop", steps=())
    if plan_type == "singlehop":
        raise ValueError("PLAN requires sequential or comparison plan_type")
    if not 2 <= len(steps_value) <= max_steps:
        raise ValueError(f"PLAN requires between 2 and {max_steps} retrieval steps")

    steps: list[TeacherPlanStep] = []
    for expected_step_id, step_value in enumerate(steps_value, start=1):
        if not isinstance(step_value, dict):
            raise ValueError("each teacher plan step must be a JSON object")
        if set(step_value) != set(TEACHER_PLAN_STEP_FIELDS):
            raise ValueError(
                "teacher plan step fields must be exactly "
                f"{list(TEACHER_PLAN_STEP_FIELDS)}"
            )
        step_id = step_value["step_id"]
        depends_on = step_value["depends_on"]
        if isinstance(step_id, bool) or not isinstance(step_id, int):
            raise ValueError("teacher plan step_id must be an integer")
        if step_id != expected_step_id:
            raise ValueError("teacher plan step_id values must be consecutive from 1")
        if not isinstance(depends_on, list):
            raise ValueError("teacher plan depends_on must be a list of integers")
        normalized_dependencies: list[int] = []
        for item in depends_on:
            if isinstance(item, str) and item.strip().isdigit():
                item = int(item.strip())
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("teacher plan depends_on must be a list of integers")
            normalized_dependencies.append(item)
        depends_on = normalized_dependencies
        if len(depends_on) != len(set(depends_on)) or any(
            item < 1 or item >= step_id for item in depends_on
        ):
            raise ValueError(
                "teacher plan dependencies must be unique earlier step IDs"
            )

        normalized_fields: dict[str, str] = {}
        for field_name in ("goal", "query_template", "expected_evidence"):
            field_value = step_value[field_name]
            if not isinstance(field_value, str):
                raise ValueError(f"teacher step field {field_name!r} must be a string")
            field_value = field_value.strip()
            if not field_value:
                raise ValueError(f"teacher step field {field_name!r} must not be empty")
            if len(field_value) > max_field_chars:
                raise ValueError(f"teacher step field {field_name!r} is too long")
            normalized_fields[field_name] = field_value

        if (
            plan_type == "comparison"
            and depends_on
            and not _ANY_PLACEHOLDER.search(normalized_fields["query_template"])
        ):
            # 7B models sometimes mark the second independent comparison branch
            # as depending on step 1 even though its query is fully concrete.
            depends_on = []

        steps.append(
            TeacherPlanStep(
                step_id=step_id,
                goal=normalized_fields["goal"],
                query_template=normalized_fields["query_template"],
                expected_evidence=normalized_fields["expected_evidence"],
                depends_on=tuple(depends_on),
            )
        )

    plan = TeacherPlan(decision="PLAN", plan_type=plan_type, steps=tuple(steps))
    if question is not None:
        validate_teacher_plan_safety(question, plan)
    return plan


def _normalized_query(value: str) -> str:
    """Normalize query text for conservative equality and containment checks."""
    tokens = _WORD_PATTERN.findall(value.casefold())
    return " ".join(
        token[:-2] if token.endswith(("'s", "’s")) else token for token in tokens
    )


def extract_query_anchors(question: str) -> tuple[str, ...]:
    """Extract quoted phrases, identifiers, and capitalized entity spans."""
    anchors: list[str] = []
    for pattern in _QUOTED_PATTERNS:
        anchors.extend(match.group(1).strip() for match in pattern.finditer(question))

    matches = list(_WORD_PATTERN.finditer(question))
    anchors.extend(
        match.group(0)
        for match in matches
        if any(char.isdigit() for char in match.group(0))
    )

    index = 0
    while index < len(matches):
        token = matches[index].group(0)
        token_lower = token.casefold()
        is_capitalized = token[:1].isupper() or token.isupper()
        if not is_capitalized or (index == 0 and token_lower in _QUESTION_WORDS):
            index += 1
            continue

        entity_tokens = [token]
        cursor = index + 1
        while cursor < len(matches):
            next_token = matches[cursor].group(0)
            next_lower = next_token.casefold()
            if next_token[:1].isupper() or next_token.isupper():
                entity_tokens.append(next_token)
            elif next_lower in _ENTITY_CONNECTORS:
                entity_tokens.append(next_token)
            else:
                break
            cursor += 1
        while entity_tokens and entity_tokens[-1].casefold() in _ENTITY_CONNECTORS:
            entity_tokens.pop()
        if entity_tokens:
            anchors.append(" ".join(entity_tokens))
        index = max(index + 1, cursor)

    unique: list[str] = []
    seen: set[str] = set()
    for anchor in anchors:
        normalized = _normalized_query(anchor)
        if normalized and normalized not in seen:
            unique.append(anchor)
            seen.add(normalized)
    return tuple(unique)


def validate_teacher_plan_safety(question: str, plan: TeacherPlan) -> None:
    """Reject unsafe, non-executable, or effectively single-hop plans."""
    if not plan.should_rewrite:
        return

    question_normalized = _normalized_query(question)
    combined_queries = " ".join(step.query_template for step in plan.steps)
    missing_anchors = [
        anchor
        for anchor in extract_query_anchors(question)
        if _normalized_query(anchor) not in _normalized_query(combined_queries)
    ]
    if missing_anchors:
        raise ValueError(f"teacher plan drops protected anchors: {missing_anchors}")
    if all(
        _normalized_query(step.query_template) == question_normalized
        for step in plan.steps
    ):
        raise ValueError("PLAN must decompose rather than repeat the full question")

    has_dependency = False
    for step in plan.steps:
        query = step.query_template
        if _META_QUERY_PATTERN.search(query):
            raise ValueError(
                f"teacher step {step.step_id} contains a meta-instruction or "
                "square-bracket placeholder"
            )
        placeholder_names = set(_ANY_PLACEHOLDER.findall(query))
        expected_placeholders = {
            f"step_{dependency}_result" for dependency in step.depends_on
        }
        if placeholder_names != expected_placeholders:
            raise ValueError(
                f"teacher step {step.step_id} placeholders must exactly match "
                "depends_on"
            )
        has_dependency = has_dependency or bool(step.depends_on)

    if plan.plan_type == "sequential":
        if not has_dependency or any(not step.depends_on for step in plan.steps[1:]):
            raise ValueError(
                "sequential plans require every step after the first to depend "
                "on earlier evidence"
            )
    elif sum(not step.depends_on for step in plan.steps) < 2:
        raise ValueError("comparison plans require at least two independent fact hops")

    validate_teacher_plan_semantics(question, plan)


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    """Return whether normalized ``value`` contains one of ``terms``."""
    normalized = _normalized_query(value)
    return any(_normalized_query(term) in normalized for term in terms)


def validate_teacher_plan_semantics(question: str, plan: TeacherPlan) -> None:
    """Reject plans that omit relations explicitly requested by the question.

    This validator intentionally uses only the question and generated plan. Gold
    answers, supporting facts, and dataset evidence are not available here.
    """
    if not plan.should_rewrite:
        return

    question_lower = question.casefold()
    plan_text = " ".join(
        value
        for step in plan.steps
        for value in (step.goal, step.query_template, step.expected_evidence)
    )
    requirements = list(_RELATION_REQUIREMENTS)
    if "lived longer" in question_lower:
        requirements.extend(
            (
                (("lived longer",), _BIRTH_PLAN_TERMS),
                (("lived longer",), _DEATH_PLAN_TERMS),
            )
        )
    else:
        requirements.extend(
            (
                (_BIRTH_QUESTION_TERMS, _BIRTH_PLAN_TERMS),
                (_DEATH_QUESTION_TERMS, _DEATH_PLAN_TERMS),
            )
        )
    requirements.append((_RELEASE_QUESTION_TERMS, _RELEASE_PLAN_TERMS))

    missing_relations = [
        "/".join(plan_terms)
        for question_terms, plan_terms in requirements
        if _contains_any(question_lower, question_terms)
        and not _contains_any(plan_text, plan_terms)
    ]
    if missing_relations:
        raise ValueError(
            "teacher plan omits required question relations: "
            f"{sorted(set(missing_relations))}"
        )

    # A director-attribute comparison contains two genuine relation chains:
    # film -> director -> birth/death/citizenship. Two root-only director
    # lookups are syntactically valid but semantically incomplete.
    director_attribute_terms: tuple[str, ...] | None = None
    if "director" in question_lower:
        if _contains_any(question_lower, ("nationality", "citizenship", "country")):
            director_attribute_terms = ("nationality", "citizenship", "country")
        elif _contains_any(question_lower, _DEATH_QUESTION_TERMS):
            director_attribute_terms = _DEATH_PLAN_TERMS
        elif _contains_any(question_lower, _BIRTH_QUESTION_TERMS):
            director_attribute_terms = _BIRTH_PLAN_TERMS
    if plan.plan_type != "comparison" or director_attribute_terms is None:
        return

    root_director_steps = [
        step
        for step in plan.steps
        if not step.depends_on
        and _contains_any(
            " ".join((step.goal, step.query_template, step.expected_evidence)),
            ("director", "directed"),
        )
    ]
    dependent_attribute_steps = [
        step
        for step in plan.steps
        if step.depends_on
        and _contains_any(
            " ".join((step.goal, step.query_template, step.expected_evidence)),
            director_attribute_terms,
        )
    ]
    root_step_ids = {step.step_id for step in root_director_steps}
    covered_root_ids = {
        dependency
        for step in dependent_attribute_steps
        for dependency in step.depends_on
        if dependency in root_step_ids
    }
    if (
        len(root_director_steps) < 2
        or len(dependent_attribute_steps) < 2
        or len(covered_root_ids) < 2
    ):
        raise ValueError(
            "director-attribute comparison requires two director root hops and "
            "two dependent attribute hops"
        )


def format_teacher_guidance(plan: TeacherPlan) -> str:
    """Render a validated plan as explicitly low-trust policy guidance."""
    if not plan.should_rewrite:
        raise ValueError("KEEP plans do not produce policy guidance")
    lines = [f"Plan type: {plan.plan_type}"]
    for step in plan.steps:
        dependencies = ", ".join(str(item) for item in step.depends_on) or "none"
        lines.extend(
            (
                f"Hop {step.step_id} goal: {step.goal}",
                f"Hop {step.step_id} query template: {step.query_template}",
                f"Hop {step.step_id} expected evidence: {step.expected_evidence}",
                f"Hop {step.step_id} depends on: {dependencies}",
            )
        )
    body = "\n".join(lines)
    return f"{_GUIDANCE_PREFIX}{body}{_GUIDANCE_SUFFIX}"


def build_guidance_token_ids(
    tokenizer,
    plan: TeacherPlan,
    guidance_mode: str,
) -> list[int]:
    """Tokenize real or length-matched generic low-trust guidance."""
    actual_ids = tokenizer.encode(
        format_teacher_guidance(plan), add_special_tokens=False
    )
    if guidance_mode != "generic":
        return actual_ids

    prefix_ids = tokenizer.encode(_GUIDANCE_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_GUIDANCE_SUFFIX, add_special_tokens=False)
    body_budget = max(0, len(actual_ids) - len(prefix_ids) - len(suffix_ids))
    filler_ids = tokenizer.encode(_GENERIC_TEXT, add_special_tokens=False)
    if not filler_ids:
        raise ValueError("generic teacher guidance must tokenize to at least one token")
    repeats = (body_budget + len(filler_ids) - 1) // len(filler_ids)
    body_ids = (filler_ids * repeats)[:body_budget]
    generic_ids = prefix_ids + body_ids + suffix_ids
    return generic_ids[: len(actual_ids)]


def insert_guidance_user_message(
    tokenizer,
    prompt_ids: list[int],
    guidance_ids: list[int],
) -> list[int]:
    """Insert guidance as a Qwen ChatML user turn before assistant generation."""
    if not guidance_ids:
        return list(prompt_ids)

    assistant_prefix_ids = tokenizer.encode(
        _CHATML_ASSISTANT_PREFIX, add_special_tokens=False
    )
    if not assistant_prefix_ids or (
        len(prompt_ids) < len(assistant_prefix_ids)
        or prompt_ids[-len(assistant_prefix_ids) :] != assistant_prefix_ids
    ):
        raise ValueError(
            "guided Search-R1 prompts must end with the Qwen ChatML assistant "
            "generation prefix"
        )

    user_prefix_ids = tokenizer.encode(_CHATML_USER_PREFIX, add_special_tokens=False)
    user_suffix_ids = tokenizer.encode(_CHATML_USER_SUFFIX, add_special_tokens=False)
    prompt_without_assistant = prompt_ids[: -len(assistant_prefix_ids)]
    return (
        prompt_without_assistant
        + user_prefix_ids
        + guidance_ids
        + user_suffix_ids
        + assistant_prefix_ids
    )


def shuffled_teacher_plans(
    plans: list[TeacherPlanResult],
    sample_ids: list[str | int],
    seed: int,
) -> list[TeacherPlanResult]:
    """Return a reproducible random derangement of plans across questions."""
    if len(plans) != len(sample_ids):
        raise ValueError("teacher plans and sample IDs must align")
    if len(plans) < 2:
        raise ValueError("shuffled teacher control requires at least two questions")

    seed_payload = json.dumps(
        {"seed": seed, "sample_ids": [str(value) for value in sample_ids]},
        sort_keys=True,
        separators=(",", ":"),
    )
    derived_seed = int.from_bytes(
        hashlib.sha256(seed_payload.encode("utf-8")).digest()[:8], "big"
    )
    generator = random.Random(derived_seed)
    indices = list(range(len(plans)))
    for _ in range(100):
        generator.shuffle(indices)
        if all(
            source_index != target_index
            for target_index, source_index in enumerate(indices)
        ):
            return [plans[index] for index in indices]

    # A rotation is a deterministic derangement fallback for very small or
    # unlucky permutations.
    return plans[1:] + plans[:1]


def paired_bootstrap_ci(
    paired_differences: list[float],
    *,
    seed: int,
    num_samples: int = 2000,
) -> tuple[float, float]:
    """Compute a paired bootstrap percentile confidence interval."""
    if not paired_differences:
        return 0.0, 0.0
    if num_samples <= 0:
        raise ValueError("bootstrap num_samples must be positive")

    values = np.asarray(paired_differences, dtype=np.float64)
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_samples, dtype=np.float64)
    chunk_size = min(64, num_samples)
    for start in range(0, num_samples, chunk_size):
        stop = min(start + chunk_size, num_samples)
        indices = generator.integers(0, len(values), size=(stop - start, len(values)))
        bootstrap_means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(lower), float(upper)


def build_shadow_metrics(
    context: dict[str, Any],
    *,
    bootstrap_seed: int = 1234,
    bootstrap_samples: int = 2000,
) -> dict[str, float]:
    """Build final baseline and paired teacher-shadow metrics."""

    def safe_ratio(numerator: float, denominator: int | float) -> float:
        return numerator / denominator if denominator else 0.0

    metrics: dict[str, float] = {}
    for mode, count in context["mode_counts"].items():
        em = safe_ratio(context["mode_reward_sums"][mode], count)
        answer_hit = safe_ratio(context["mode_answer_hit_sums"][mode], count)
        final_subem = safe_ratio(
            context.get("mode_subem_sums", {}).get(mode, 0.0), count
        )
        metrics[f"planner/{mode}_EM"] = em
        metrics[f"search/{mode}_answer_hit_rate"] = answer_hit
        metrics[f"planner/{mode}_diagnostic_SubEM"] = final_subem
        metrics[f"search/{mode}_tool_call_repair_rate"] = safe_ratio(
            context.get("mode_tool_call_repair_sums", {}).get(mode, 0.0), count
        )
        metrics[f"search/{mode}_dual_query_rate"] = safe_ratio(
            context.get("mode_dual_query_sums", {}).get(mode, 0.0), count
        )
        metrics[f"planner/{mode}_controller_completion_rate"] = safe_ratio(
            context.get("mode_controller_completion_sums", {}).get(mode, 0.0),
            context.get("mode_controller_applied_sums", {}).get(mode, 0.0),
        )
        metrics[f"search/{mode}_controller_fallback_query_rate"] = safe_ratio(
            context.get("mode_controller_fallback_query_sums", {}).get(mode, 0.0),
            context.get("mode_controller_step_sums", {}).get(mode, 0.0),
        )
        metrics[f"search/{mode}_controller_dependent_fallback_rate"] = safe_ratio(
            context.get("mode_controller_fallback_query_sums", {}).get(mode, 0.0),
            context.get("mode_controller_dependent_step_sums", {}).get(mode, 0.0),
        )
        metrics[f"search/{mode}_dependent_query_binding_valid_rate"] = safe_ratio(
            context.get("mode_controller_binding_valid_sums", {}).get(mode, 0.0),
            context.get("mode_controller_dependent_step_sums", {}).get(mode, 0.0),
        )
        metrics[f"search/{mode}_binding_attempts_per_dependent_hop"] = safe_ratio(
            context.get("mode_controller_binding_attempt_sums", {}).get(mode, 0.0),
            context.get("mode_controller_dependent_step_sums", {}).get(mode, 0.0),
        )
        metrics[f"search/{mode}_binding_alias_rate"] = safe_ratio(
            context.get("mode_controller_binding_alias_sums", {}).get(mode, 0.0),
            context.get("mode_controller_binding_valid_sums", {}).get(mode, 0.0),
        )
        metrics[f"search/{mode}_unresolved_placeholder_rate"] = safe_ratio(
            context.get("mode_unresolved_placeholder_sums", {}).get(mode, 0.0),
            context.get("mode_controller_step_sums", {}).get(mode, 0.0),
        )
        failure_reason_counts = context.get("mode_binding_failure_reason_sums", {}).get(
            mode, {}
        )
        failure_reasons = {
            "premature_answer",
            "unresolved_placeholder",
            "empty_query",
            "ungrounded_value",
            "unbound_query",
            "parser_failure",
            *failure_reason_counts,
        }
        for reason in sorted(failure_reasons):
            metrics[f"search/{mode}_binding_failure_{reason}_rate"] = safe_ratio(
                failure_reason_counts.get(reason, 0.0),
                context.get("mode_controller_binding_attempt_sums", {}).get(mode, 0.0),
            )
        metrics[f"search/{mode}_average_search_count"] = safe_ratio(
            context.get("mode_search_sums", {}).get(mode, 0.0), count
        )
        metrics[f"planner/{mode}_average_turn_count"] = safe_ratio(
            context.get("mode_turn_sums", {}).get(mode, 0.0), count
        )
        metrics[f"planner/{mode}_average_generated_token_count"] = safe_ratio(
            context.get("mode_generated_token_sums", {}).get(mode, 0.0), count
        )
        metrics[f"planner/{mode}_synthesis_format_valid_rate"] = safe_ratio(
            context.get("mode_synthesis_format_valid_sums", {}).get(mode, 0.0),
            context.get("mode_controller_applied_sums", {}).get(mode, 0.0),
        )
        metrics[f"planner/{mode}_synthesis_format_repair_rate"] = safe_ratio(
            context.get("mode_synthesis_format_repair_sums", {}).get(mode, 0.0),
            context.get("mode_controller_applied_sums", {}).get(mode, 0.0),
        )
        if mode == "unguided":
            metrics["eval/unguided_EM"] = em

    for mode, count in context["paired_counts"].items():
        metrics[f"planner/{mode}_minus_unguided"] = safe_ratio(
            context["paired_uplift_sums"][mode], count
        )
        metrics[f"planner/{mode}_answer_hit_delta"] = safe_ratio(
            context["paired_answer_hit_sums"][mode], count
        )
        metrics[f"planner/{mode}_query_change_rate"] = safe_ratio(
            context["query_change_sums"][mode],
            context["query_change_counts"][mode],
        )
        paired_values = context.get("paired_uplifts", {}).get(mode, [])
        if paired_values:
            lower, upper = paired_bootstrap_ci(
                paired_values,
                seed=bootstrap_seed,
                num_samples=bootstrap_samples,
            )
            metrics[f"planner/{mode}_uplift_ci_low"] = lower
            metrics[f"planner/{mode}_uplift_ci_high"] = upper

    plan_valid_values = list(context["plan_valid_by_id"].values())
    metrics["planner/plan_valid_rate"] = safe_ratio(
        sum(plan_valid_values), len(plan_valid_values)
    )
    cache_hit_values = list(context["plan_cache_hit_by_id"].values())
    metrics["planner/cache_hit_rate"] = safe_ratio(
        sum(cache_hit_values), len(cache_hit_values)
    )
    decisions = list(context.get("plan_decision_by_id", {}).values())
    metrics["planner/rewrite_rate"] = safe_ratio(
        sum(decision in {"PLAN", "REWRITE"} for decision in decisions),
        len(decisions),
    )
    metrics["planner/query_change_rate"] = metrics.get(
        "planner/guided_query_change_rate", 0.0
    )
    metrics["planner/answer_hit_delta"] = metrics.get(
        "planner/guided_answer_hit_delta", 0.0
    )
    return metrics


class TeacherPlanCache:
    """Process-local cache with optional atomic filesystem persistence."""

    def __init__(self, cache_dir: str | None):
        self._memory: dict[str, TeacherPlanResult] = {}
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None

    def get(self, key: str) -> TeacherPlanResult | None:
        """Load a cached plan result if it exists."""
        if key in self._memory:
            return replace(self._memory[key], cache_hit=True)
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        result = TeacherPlanResult.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        self._memory[key] = replace(result, cache_hit=False)
        return replace(result, cache_hit=True)

    def put(self, key: str, result: TeacherPlanResult) -> None:
        """Cache a result and atomically persist it when configured."""
        self._memory[key] = replace(result, cache_hit=False)
        if self._cache_dir is None:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / f"{key}.json"
        temporary_path = self._cache_dir / (f".{key}.{os.getpid()}.{uuid4().hex}.tmp")
        temporary_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)


class FrozenTeacherPlanner:
    """Client for the independent frozen teacher-planner rollout group."""

    def __init__(self, cfg: DictConfig, tokenizer):
        teacher_cfg = cfg.teacher_planner
        self.tokenizer = tokenizer
        self.teacher_version = str(teacher_cfg.version)
        self.seed = int(teacher_cfg.get("seed", cfg.data.get("seed", 0)))
        self.max_field_chars = int(teacher_cfg.get("max_field_chars", 512))
        self.max_plan_chars = int(teacher_cfg.get("max_plan_chars", 6144))
        self.max_steps = int(teacher_cfg.get("max_steps", 8))
        self.max_attempts = int(teacher_cfg.get("max_attempts", 3))
        self.require_plan = bool(teacher_cfg.get("require_plan", False))
        if self.max_steps < 2:
            raise ValueError("teacher_planner.max_steps must be at least 2")
        if self.max_attempts < 1:
            raise ValueError("teacher_planner.max_attempts must be positive")
        self.cache_only = bool(teacher_cfg.get("cache_only", False))
        if self.cache_only and not teacher_cfg.get("cache_dir"):
            raise ValueError("teacher_planner.cache_only requires cache_dir")
        self.cache = TeacherPlanCache(teacher_cfg.get("cache_dir"))
        self.sampling_params = {
            "temperature": 0.0,
            "max_new_tokens": int(teacher_cfg.get("max_new_tokens", 768)),
        }

    def build_prompt_ids(self, question: str) -> list[int]:
        """Build a teacher request containing the question and no label data."""
        messages = [
            {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    def build_repair_prompt_ids(
        self, question: str, raw_response: str, error: str
    ) -> list[int]:
        """Build a question-only repair request after validation fails."""
        repair_request = (
            f"Question: {question}\n\n"
            "The previous plan was rejected by the deterministic validator. "
            "Return a corrected JSON plan that follows every system rule. Do not "
            "copy the validator message into the JSON and do not answer the question.\n"
            f"Validation error: {error}\n"
            f"Rejected plan: {raw_response[: self.max_plan_chars]}"
        )
        messages = [
            {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": repair_request},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    def cache_response(self, question: str, raw_response: str) -> TeacherPlanResult:
        """Validate and persist one teacher response for ``question``."""
        plan_id = teacher_plan_cache_key(question, self.teacher_version, self.seed)
        try:
            plan = parse_teacher_plan(
                raw_response,
                question=question,
                max_field_chars=self.max_field_chars,
                max_plan_chars=self.max_plan_chars,
                max_steps=self.max_steps,
            )
            if self.require_plan and not plan.should_plan:
                raise ValueError("this evaluation requires a multi-hop PLAN")
            result = TeacherPlanResult(
                plan_id=plan_id,
                valid=True,
                plan=plan,
                raw_response=raw_response,
            )
        except ValueError as error:
            result = TeacherPlanResult(
                plan_id=plan_id,
                valid=False,
                plan=None,
                raw_response=raw_response,
                error=str(error),
            )
        self.cache.put(plan_id, result)
        return result

    async def get_plan(
        self,
        question: str,
        generate: Callable[..., Awaitable[dict[str, Any]]],
    ) -> TeacherPlanResult:
        """Return a cached plan or generate and strictly validate a new one."""
        plan_id = teacher_plan_cache_key(question, self.teacher_version, self.seed)
        cached = self.cache.get(plan_id)
        if cached is not None:
            return cached
        if self.cache_only:
            raise RuntimeError(
                "teacher plan cache miss in cache-only mode: "
                f"plan_id={plan_id}, question={question[:120]!r}"
            )

        prompt_ids = self.build_prompt_ids(question)
        result: TeacherPlanResult | None = None
        for _ in range(self.max_attempts):
            generate_result = await generate(
                prompt_ids,
                sampling_params=self.sampling_params,
                rollout_name=TEACHER_PLANNER_ROLLOUT_NAME,
            )
            raw_response = self.tokenizer.decode(
                generate_result["output_ids"], skip_special_tokens=True
            )
            result = self.cache_response(question, raw_response)
            if result.valid:
                return result
            prompt_ids = self.build_repair_prompt_ids(
                question, raw_response, result.error or "unknown validation error"
            )
        assert result is not None
        return result
