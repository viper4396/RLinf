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

"""Label-only diagnostics for frozen Search-R1 teacher evaluation."""

import re
from collections import defaultdict
from typing import Any

from rlinf.agents.searchr1.teacher_planner import (
    TeacherPlan,
    paired_bootstrap_ci,
)
from rlinf.algorithms.searchr1_scoring import subem_check

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_RELATION_ALIASES = (
    (("date of birth", "birth date"), ("date of birth", "birth date", "born")),
    (("place of birth",), ("place of birth", "birthplace", "born in")),
    (("date of death", "death date"), ("date of death", "death date", "died")),
    (("place of death",), ("place of death", "death place", "died in")),
    (
        ("publication date", "release date", "release year"),
        ("publication date", "release date", "release year", "came out", "premiere"),
    ),
    (
        ("country of citizenship", "nationality"),
        ("country of citizenship", "nationality", "citizenship", "country"),
    ),
    (
        ("country of origin",),
        ("country of origin", "origin country", "nationality", "country"),
    ),
    (("director",), ("director", "directed")),
    (("performer",), ("performer", "actor", "actress", "starred")),
    (("educated at",), ("educated", "education", "alma mater", "university")),
    (("employer",), ("employer", "worked for", "employed")),
    (("inception",), ("inception", "founded", "established")),
    (("place of burial",), ("place of burial", "burial", "buried")),
    (("founded by",), ("founded by", "founder")),
    (("occupation",), ("occupation", "profession")),
    (("award received",), ("award received", "award")),
    (("cause of death",), ("cause of death", "death cause")),
    (("presenter",), ("presenter", "host")),
    (("spouse",), ("spouse", "wife", "husband", "married")),
    (("father",), ("father",)),
    (("mother",), ("mother",)),
)


def _normalize(value: Any) -> str:
    """Return a case-insensitive, punctuation-insensitive token sequence."""
    return " ".join(_TOKEN_PATTERN.findall(str(value).casefold()))


def _contains_phrase(container: str, candidate: str) -> bool:
    """Return whether a normalized phrase occurs with token boundaries."""
    normalized_container = f" {_normalize(container)} "
    normalized_candidate = _normalize(candidate)
    return bool(normalized_candidate) and (
        f" {normalized_candidate} " in normalized_container
    )


def _relation_terms(relation: str) -> tuple[str, ...]:
    """Return deterministic aliases for one 2Wiki evidence relation."""
    normalized_relation = _normalize(relation)
    for source_terms, plan_terms in _RELATION_ALIASES:
        if any(_normalize(term) in normalized_relation for term in source_terms):
            return plan_terms
    return (relation,)


def audit_plan_semantic_coverage(
    plan_value: dict[str, Any] | TeacherPlan | None,
    evidences: list[list[Any]] | tuple[tuple[Any, ...], ...],
) -> tuple[bool, float, list[int]]:
    """Audit whether a plan DAG covers every gold evidence relation.

    This function is strictly offline: it consumes dataset evidence labels only
    after trajectories have finished. It must never be called by plan selection,
    query construction, retrieval, or synthesis code.

    Args:
        plan_value: Serialized or parsed teacher plan.
        evidences: Gold ``[subject, relation, object]`` triples.

    Returns:
        A tuple of full-plan coverage, covered-edge fraction, and uncovered
        evidence indices.
    """
    triples = [tuple(evidence[:3]) for evidence in evidences if len(evidence) >= 3]
    if not triples:
        return True, 1.0, []
    if plan_value is None:
        return False, 0.0, list(range(len(triples)))
    try:
        plan = (
            plan_value
            if isinstance(plan_value, TeacherPlan)
            else TeacherPlan.from_dict(plan_value)
        )
    except (KeyError, TypeError, ValueError):
        return False, 0.0, list(range(len(triples)))
    if not plan.should_plan:
        return False, 0.0, list(range(len(triples)))

    step_text = {
        step.step_id: " ".join((step.goal, step.query_template, step.expected_evidence))
        for step in plan.steps
    }
    used_steps: set[int] = set()
    matched_edges: dict[int, int] = {}
    object_producers: dict[str, set[int]] = defaultdict(set)

    # Evidence triples are normally topological, but repeat the pass so the
    # audit remains correct when independent comparison branches are interleaved.
    progress = True
    while progress:
        progress = False
        for edge_index, (subject, relation, object_value) in enumerate(triples):
            if edge_index in matched_edges:
                continue
            relation_terms = _relation_terms(str(relation))
            subject_producers = object_producers.get(_normalize(subject), set())
            for step in plan.steps:
                if step.step_id in used_steps:
                    continue
                text = step_text[step.step_id]
                if not any(_contains_phrase(text, term) for term in relation_terms):
                    continue
                literal_subject = _contains_phrase(text, str(subject))
                dependent_subject = any(
                    producer in step.depends_on
                    and f"{{step_{producer}_result}}" in step.query_template
                    for producer in subject_producers
                )
                if not literal_subject and not dependent_subject:
                    continue
                matched_edges[edge_index] = step.step_id
                used_steps.add(step.step_id)
                object_producers[_normalize(object_value)].add(step.step_id)
                progress = True
                break

    uncovered = [index for index in range(len(triples)) if index not in matched_edges]
    edge_coverage = len(matched_edges) / len(triples)
    return not uncovered, edge_coverage, uncovered


def _dataset_record(records: list[dict[str, Any]], sample_id: Any) -> dict[str, Any]:
    """Resolve a rollout sample ID against the post-filter validation records."""
    try:
        index = int(sample_id)
    except (TypeError, ValueError):
        return {}
    if index < 0 or index >= len(records):
        return {}
    return records[index]


def build_label_only_diagnostics(
    results: list[dict[str, Any]],
    dataset_records: list[dict[str, Any]],
    *,
    bootstrap_seed: int = 1234,
    bootstrap_samples: int = 2000,
) -> dict[str, float]:
    """Attach offline labels and compute plan/type diagnostics.

    The input results already contain completed trajectories. Dataset evidence
    and question-type labels are joined here, on the runner, and are never sent
    back to an agent-loop worker or inference service.
    """
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_audits: dict[str, tuple[bool, float, list[int]]] = {}
    for result in results:
        record = _dataset_record(dataset_records, result.get("sample_id"))
        question_type = str(record.get("question_type") or "unknown")
        evidences = record.get("evidences") or []
        result["query_id"] = record.get("query_id")
        result["question_type"] = question_type
        result["supporting_facts"] = record.get("supporting_facts") or []
        result["evidences"] = evidences

        visible_evidence = "\n".join(
            str(turn.get("visible_evidence") or "") for turn in result.get("turns", [])
        )
        evidence_objects = [evidence[2] for evidence in evidences if len(evidence) >= 3]
        covered_objects = sum(
            int(subem_check(visible_evidence, evidence_object))
            for evidence_object in evidence_objects
        )
        result["gold_evidence_object_coverage"] = (
            covered_objects / len(evidence_objects) if evidence_objects else 1.0
        )
        result["gold_evidence_full_chain"] = bool(
            covered_objects == len(evidence_objects)
        )

        sample_key = str(result.get("sample_id"))
        by_sample[sample_key].append(result)
        if result.get("guidance_mode") == "guided" and sample_key not in sample_audits:
            sample_audits[sample_key] = audit_plan_semantic_coverage(
                result.get("teacher_plan"), evidences
            )

    for sample_key, sample_results in by_sample.items():
        audit = sample_audits.get(sample_key, (False, 0.0, []))
        for result in sample_results:
            result["plan_semantic_coverage"] = audit[0]
            result["plan_semantic_edge_coverage"] = audit[1]
            result["plan_uncovered_evidence_indices"] = audit[2]

    metrics: dict[str, float] = {}
    metrics["planner/plan_semantic_coverage_rate"] = (
        sum(int(audit[0]) for audit in sample_audits.values()) / len(sample_audits)
        if sample_audits
        else 0.0
    )
    metrics["planner/plan_semantic_edge_coverage_rate"] = (
        sum(audit[1] for audit in sample_audits.values()) / len(sample_audits)
        if sample_audits
        else 0.0
    )

    question_types = sorted(
        {
            str(result.get("question_type") or "unknown")
            for result in results
            if result.get("question_type")
        }
    )
    type_groups = {question_type: {question_type} for question_type in question_types}
    type_groups["compositional_inference"] = {"compositional", "inference"}
    for group_name, included_types in type_groups.items():
        group_results = [
            result
            for result in results
            if result.get("question_type") in included_types
        ]
        modes = sorted({str(result.get("guidance_mode")) for result in group_results})
        for mode in modes:
            mode_results = [
                result
                for result in group_results
                if str(result.get("guidance_mode")) == mode
            ]
            if not mode_results:
                continue
            prefix = f"type/{group_name}/{mode}"
            metrics[f"planner/{prefix}_EM"] = sum(
                float(result.get("reward", 0.0)) for result in mode_results
            ) / len(mode_results)
            metrics[f"search/{prefix}_answer_hit_rate"] = sum(
                float(bool(result.get("answer_hit", False))) for result in mode_results
            ) / len(mode_results)
            metrics[f"search/{prefix}_gold_evidence_object_coverage"] = sum(
                float(result.get("gold_evidence_object_coverage", 0.0))
                for result in mode_results
            ) / len(mode_results)
            metrics[f"search/{prefix}_gold_evidence_full_chain_rate"] = sum(
                float(bool(result.get("gold_evidence_full_chain", False)))
                for result in mode_results
            ) / len(mode_results)

        paired_differences: list[float] = []
        for sample_results in by_sample.values():
            if not sample_results or sample_results[0].get("question_type") not in (
                included_types
            ):
                continue
            guided = [
                float(result.get("reward", 0.0))
                for result in sample_results
                if result.get("guidance_mode") == "guided"
            ]
            unguided = [
                float(result.get("reward", 0.0))
                for result in sample_results
                if result.get("guidance_mode") == "unguided"
            ]
            if guided and unguided:
                paired_differences.append(
                    sum(guided) / len(guided) - sum(unguided) / len(unguided)
                )
        if paired_differences:
            metric_prefix = f"planner/type/{group_name}"
            metrics[f"{metric_prefix}/guided_minus_unguided"] = sum(
                paired_differences
            ) / len(paired_differences)
            ci_low, ci_high = paired_bootstrap_ci(
                paired_differences,
                seed=bootstrap_seed,
                num_samples=bootstrap_samples,
            )
            metrics[f"{metric_prefix}/uplift_ci_low"] = ci_low
            metrics[f"{metric_prefix}/uplift_ci_high"] = ci_high
    return metrics


def build_abc_acceptance_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Evaluate the frozen stage-two A/B/C promotion gates."""

    def passed(value: bool) -> float:
        return float(value)

    checks = {
        "acceptance/A_plan_valid": metrics.get("planner/plan_valid_rate", 0.0) >= 0.99,
        "acceptance/A_cache_hit": metrics.get("planner/cache_hit_rate", 0.0) == 1.0,
        "acceptance/A_controller_completion": metrics.get(
            "planner/guided_controller_completion_rate", 0.0
        )
        >= 0.99,
        "acceptance/A_synthesis_format": metrics.get(
            "planner/guided_synthesis_format_valid_rate", 0.0
        )
        >= 0.99,
        "acceptance/A_unresolved_placeholder": metrics.get(
            "search/guided_unresolved_placeholder_rate", 1.0
        )
        == 0.0,
        "acceptance/B_plan_semantic_coverage": metrics.get(
            "planner/plan_semantic_coverage_rate", 0.0
        )
        >= 0.95,
        "acceptance/B_dependent_binding": metrics.get(
            "search/guided_dependent_query_binding_valid_rate", 0.0
        )
        >= 0.95,
        "acceptance/B_dependent_fallback": metrics.get(
            "search/guided_controller_dependent_fallback_rate", 1.0
        )
        <= 0.10,
        "acceptance/B_sequential_uplift": metrics.get(
            "planner/type/compositional_inference/guided_minus_unguided", -1.0
        )
        >= 0.01,
        "acceptance/B_sequential_ci": metrics.get(
            "planner/type/compositional_inference/uplift_ci_low", -1.0
        )
        >= 0.0,
        "acceptance/B_no_type_regression": all(
            metrics.get(f"planner/type/{question_type}/guided_minus_unguided", -1.0)
            >= -0.01
            for question_type in (
                "compositional",
                "inference",
                "comparison",
                "bridge_comparison",
            )
        ),
        "acceptance/C_overall_uplift": metrics.get(
            "planner/guided_minus_unguided", -1.0
        )
        >= 0.02,
        "acceptance/C_overall_ci": metrics.get("planner/guided_uplift_ci_low", -1.0)
        > 0.0,
    }
    output = {name: passed(value) for name, value in checks.items()}
    output["acceptance/A_pass"] = passed(
        all(value for name, value in checks.items() if name.startswith("acceptance/A_"))
    )
    output["acceptance/B_pass"] = passed(
        all(value for name, value in checks.items() if name.startswith("acceptance/B_"))
    )
    output["acceptance/C_pass"] = passed(
        all(value for name, value in checks.items() if name.startswith("acceptance/C_"))
    )
    output["acceptance/ABC_pass"] = passed(
        bool(output["acceptance/A_pass"])
        and bool(output["acceptance/B_pass"])
        and bool(output["acceptance/C_pass"])
    )
    return output
