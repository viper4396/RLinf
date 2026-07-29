# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic render and completion audit for graph-memory v2."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.item import (
    item_terminal_facts,
    item_value,
)
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    AuditResult,
    EvidenceKind,
    EvidenceStatus,
    GapReport,
    GraphEventType,
    PayloadNode,
    RenderNode,
    TaskContract,
)


class RenderError(RuntimeError):
    """Raised when deterministic rendering is attempted before audit passes."""


@dataclass(frozen=True)
class RenderValidation:
    """Mechanical Phase 4 format validation result."""

    valid: bool
    code: str = "OK"
    message: str = ""
    row_count: int = 0
    columns: tuple[str, ...] = ()
    outside_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _terminal_facts(runtime: Any) -> list[Any]:
    if runtime.config.item_require_terminal_fact:
        return item_terminal_facts(runtime)
    graph = runtime.evidence_graph
    contract = runtime.contract
    facts = [
        node
        for node in graph.iter_kind(EvidenceKind.FACT)
        if node.status == EvidenceStatus.VERIFIED
    ]
    if contract is None:
        return []
    if contract.terminal_fact_ref:
        terminal = runtime.evidence_graph.nodes.get(contract.terminal_fact_ref)
        if terminal is None:
            terminal = runtime.evidence_graph.get_by_canonical(
                contract.terminal_fact_ref, EvidenceKind.FACT
            )
        return [
            node
            for node in facts
            if terminal is not None and node.node_id == terminal.node_id
        ]
    if contract.terminal_predicate:
        facts = [
            node
            for node in facts
            if node.payload.get("predicate") == contract.terminal_predicate
        ]
    tagged = [
        node
        for node in facts
        if "terminal" in node.tags or node.payload.get("terminal")
    ]
    if tagged:
        return tagged
    return facts


def audit_item(runtime: Any) -> AuditResult:
    """Check terminal Fact, provenance, uniqueness, and open conflicts."""

    contract: TaskContract | None = runtime.contract
    if contract is None:
        return AuditResult(
            passed=False,
            status="INCOMPLETE",
            gap_report=GapReport(missing_fields=("task_contract",)),
        )
    if contract.answer_kind != "item":
        return AuditResult(
            passed=False,
            status="INCOMPLETE",
            gap_report=GapReport(missing_fields=("answer_kind:item",)),
        )
    facts = _terminal_facts(runtime)
    missing: list[str] = []
    if not facts:
        missing.append("terminal_fact")
    if len(facts) > 1:
        missing.append("unique_terminal_fact")
    open_conflicts = tuple(
        node.node_id
        for node in runtime.evidence_graph.iter_kind(EvidenceKind.CONFLICT)
        if node.status in {EvidenceStatus.OPEN, EvidenceStatus.CONFLICTED}
    )
    if open_conflicts:
        missing.append("no_open_conflict")

    provenance_missing = []
    for fact in facts:
        claim_ref = fact.payload.get("claim_ref")
        claim = runtime.evidence_graph.nodes.get(claim_ref)
        if claim is None:
            provenance_missing.append(fact.node_id)
            continue
        source_ids: set[str] = set()
        invalid_source = False
        pending = [claim.node_id]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            for edge in runtime.evidence_graph.incoming(current):
                source = runtime.evidence_graph.nodes.get(edge.source_id)
                if source is None:
                    continue
                if source.kind == EvidenceKind.SOURCE:
                    if (
                        isinstance(source.payload.get("uri"), str)
                        and source.payload.get("uri", "").strip()
                        and source.payload.get("locator") is not None
                        and str(source.payload.get("locator")).strip()
                    ):
                        source_ids.add(source.node_id)
                    else:
                        invalid_source = True
                elif edge.relation in {
                    "SUPPORTS",
                    "OBSERVED_IN",
                    "NORMALIZES_TO",
                    "DERIVED_FROM",
                }:
                    pending.append(source.node_id)
            for edge in runtime.evidence_graph.outgoing(current, "OBSERVED_IN"):
                source = runtime.evidence_graph.nodes.get(edge.target_id)
                if source is not None and source.kind == EvidenceKind.SOURCE:
                    if (
                        isinstance(source.payload.get("uri"), str)
                        and source.payload.get("uri", "").strip()
                        and source.payload.get("locator") is not None
                        and str(source.payload.get("locator")).strip()
                    ):
                        source_ids.add(source.node_id)
                    else:
                        invalid_source = True
        source_count = len(source_ids)
        if (
            contract.citation_policy.require_source
            and source_count < contract.citation_policy.min_independent_sources
        ):
            provenance_missing.append(fact.node_id)
        elif invalid_source:
            provenance_missing.append(fact.node_id)
    if provenance_missing:
        missing.append("required_provenance")

    if contract is not None:
        missing_value = [
            fact.node_id
            for fact in facts
            if fact.payload.get(
                contract.final_value_field,
                fact.payload.get("value", fact.payload.get("object")),
            )
            in (None, "")
        ]
        if missing_value:
            missing.append("terminal_value")

    passed = not missing and not open_conflicts
    return AuditResult(
        passed=passed,
        status="PASS" if passed else "INCOMPLETE",
        fact_refs=tuple(fact.node_id for fact in facts),
        gap_report=GapReport(
            missing_fields=tuple(missing),
            open_conflicts=open_conflicts,
            suggested_action_templates=("resolve_terminal_fact",) if not passed else (),
        ),
    )


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def render_item(runtime: Any) -> str:
    """Render exactly one accepted Fact as a Markdown pipe table."""

    audit = audit_item(runtime)
    if not audit.passed:
        raise RenderError(f"Cannot render incomplete item graph: {audit.gap_report}")
    fact = runtime.evidence_graph.nodes[audit.fact_refs[0]]
    contract = runtime.contract
    assert contract is not None
    value = fact.payload.get(
        contract.final_value_field,
        fact.payload.get("value", fact.payload.get("object", "")),
    )
    header = contract.output_columns[0] if contract.output_columns else "Item"
    return f"```markdown\n| {_escape(header)} |\n| :--- |\n| {_escape(value)} |\n```"


def render(runtime: Any) -> str:
    """Dispatch deterministic rendering by the compiled Task Contract."""

    if runtime.contract is None:
        raise RenderError("No Task Contract has been compiled")
    if runtime.contract.answer_kind == "item":
        return render_item(runtime)
    raise RenderError(
        f"Phase 1 renderer only supports item, got {runtime.contract.answer_kind!r}"
    )


def _render_node_card(runtime: Any, node: Any) -> dict[str, Any]:
    """Return an in-scope, provenance-preserving render card."""

    card = {
        "node_id": node.node_id,
        "kind": node.kind.value,
        "canonical_key": node.canonical_key,
        "status": node.status.value,
        "payload": copy.deepcopy(node.payload),
        "provenance": {
            "created_by_action": node.created_by_action,
            "created_by_sub_traj": node.created_by_sub_traj,
            "proposed_by_role": node.proposed_by_role,
            "proposed_by_turn": node.proposed_by_turn,
            "tool_result_refs": list(node.tool_result_refs),
        },
    }
    if node.kind == EvidenceKind.SOURCE:
        for key in ("content", "text", "raw_text", "excerpt"):
            if key in card["payload"]:
                value = str(card["payload"][key]).split()
                card["payload"][key] = " ".join(
                    value[: runtime.config.max_source_excerpt_tokens]
                )
    return card


def build_render_payload(runtime: Any) -> dict[str, Any]:
    """Build the Phase 4 logical render payload and deterministic pages.

    Only active Entity/Fact nodes and one-hop provenance Sources are included.
    Candidate details are intentionally reduced to provenance references.
    """

    graph = runtime.evidence_graph
    facts = sorted(
        (
            node
            for node in graph.iter_kind(EvidenceKind.FACT)
            if node.status
            not in {
                EvidenceStatus.DISPUTED,
                EvidenceStatus.INVALIDATED,
                EvidenceStatus.REFUTED,
                EvidenceStatus.REJECTED,
                EvidenceStatus.RETIRED,
            }
        ),
        key=lambda node: node.node_id,
    )
    if runtime.answer_type == "item" and runtime.config.item_require_terminal_fact:
        facts = item_terminal_facts(runtime)
    entities: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    allowed_refs: set[str] = set()
    rows: list[dict[str, Any]] = []
    for fact in facts:
        allowed_refs.add(fact.node_id)
        subject_ref = str(fact.payload.get("subject_ref", ""))
        subject = graph.nodes.get(subject_ref)
        if (
            subject is not None
            and subject.active
            and subject.kind == EvidenceKind.ENTITY
        ):
            entities[subject.node_id] = _render_node_card(runtime, subject)
            allowed_refs.add(subject.node_id)
        source_ids = {str(ref) for ref in fact.payload.get("source_refs", ())}
        source_ids.update(
            edge.target_id
            for edge in graph.outgoing(fact.node_id, "SUPPORTED_BY")
            if graph.nodes.get(edge.target_id)
            and graph.nodes[edge.target_id].kind == EvidenceKind.SOURCE
        )
        for source_id in sorted(source_ids):
            source = graph.nodes.get(source_id)
            if (
                source is not None
                and source.active
                and source.kind == EvidenceKind.SOURCE
            ):
                sources[source.node_id] = _render_node_card(runtime, source)
                allowed_refs.add(source.node_id)
        value = (
            item_value(runtime, fact)
            if runtime.answer_type == "item"
            else fact.payload.get(
                runtime.format_requirements.get("value_field", "value"),
                fact.payload.get("value", fact.payload.get("object", "")),
            )
        )
        rows.append(
            {
                "fact_ref": fact.node_id,
                "subject_ref": subject.node_id if subject is not None else subject_ref,
                "subject": (
                    subject.payload.get("canonical_name", subject.canonical_key)
                    if subject is not None
                    else subject_ref
                ),
                "predicate": fact.payload.get("predicate", ""),
                "value": value,
                "source_refs": sorted(source_ids),
            }
        )
    output_columns = runtime.format_requirements.get("columns")
    if not output_columns:
        output_columns = runtime.format_requirements.get("output_columns")
    if not output_columns:
        output_columns = ["Item"] if runtime.answer_type == "item" else ["Item"]
    output_columns = [str(column) for column in output_columns]
    page_size = max(1, int(runtime.config.max_render_page_rows))
    pages: list[list[dict[str, Any]]] = []
    current_page: list[dict[str, Any]] = []
    page_budget = max(1, runtime.config.max_payload_tokens - 256)
    for row in rows:
        candidate_page = [*current_page, row]
        over_tokens = (
            bool(current_page)
            and len(json.dumps(candidate_page, ensure_ascii=False, default=str)) // 4
            > page_budget
        )
        if len(current_page) >= page_size or over_tokens:
            pages.append(current_page)
            current_page = [row]
        else:
            current_page = candidate_page
    if current_page:
        pages.append(current_page)
    if not pages:
        pages = [[]]
    if not pages:
        pages = [[]]
    payload = {
        "event": "RENDER_REQUIRED",
        "graph_version": runtime.version,
        "question": runtime.question,
        "answer_type": runtime.answer_type,
        "columns": output_columns,
        "order": copy.deepcopy(
            runtime.format_requirements.get("order", "graph_node_id")
        ),
        "missing_value_policy": runtime.format_requirements.get(
            "missing_value_policy", "include a blank cell"
        ),
        "markdown": {"fenced": True, "table_required": True},
        "entities": list(entities.values()),
        "facts": [_render_node_card(runtime, fact) for fact in facts],
        "sources": list(sources.values()),
        "rows": rows,
        "pages": pages,
        "allowed_refs": sorted(allowed_refs),
        "excluded_kinds": ["candidate", "claim", "conflict", "retired", "disputed"],
    }
    return payload


def start_render(runtime: Any) -> dict[str, Any]:
    """Create a Render node and immutable physical payload pages."""

    runtime.render_attempt += 1
    runtime.workflow_phase = "render"
    runtime.render_page_index = 0
    runtime.render_page_answers = []
    payload = build_render_payload(runtime)
    runtime.render_payload = payload
    pages = payload["pages"]
    logical_id = f"payload:render:{runtime.render_attempt}"
    payload_ids: list[str] = []
    for page_index, rows in enumerate(pages):
        payload_id = f"{logical_id}:page:{page_index + 1}"
        page_payload = {
            "event": payload["event"],
            "graph_version": payload["graph_version"],
            "question": payload["question"],
            "answer_type": payload["answer_type"],
            "columns": payload["columns"],
            "order": payload["order"],
            "missing_value_policy": payload["missing_value_policy"],
            "markdown": payload["markdown"],
            "page_index": page_index,
            "page_count": len(pages),
            "rows": rows,
            "allowed_refs": payload["allowed_refs"],
        }
        runtime.activation_dag.add_node(
            PayloadNode(
                payload_id=payload_id,
                selector={"render": True},
                projection={"active_entity_fact_provenance": True},
                max_tokens=runtime.config.max_payload_tokens,
                required=True,
                graph_version=runtime.version,
                token_count=max(
                    1,
                    len(json.dumps(page_payload, ensure_ascii=False, default=str)) // 4,
                ),
                retrieval_metadata={
                    "phase": "render",
                    "logical_payload_id": logical_id,
                    "page_index": page_index,
                    "page_count": len(pages),
                },
                body=page_payload,
                page_index=page_index,
                page_count=len(pages),
            )
        )
        payload_ids.append(payload_id)
    runtime.render_payload_ids = tuple(payload_ids)
    render_id = f"render:{runtime.render_attempt}"
    runtime.activation_dag.add_node(
        RenderNode(
            render_id=render_id,
            answer_kind=runtime.answer_type,
            state="required",
            attempt=runtime.render_attempt,
            graph_version=runtime.version,
            payload_ids=tuple(payload_ids),
            page_count=len(payload_ids),
        )
    )
    runtime._append_event(
        GraphEventType.CREATE_RENDER,
        actor_role="system",
        node_ids=(render_id, *payload_ids),
        payload={
            "attempt": runtime.render_attempt,
            "graph_version": runtime.version,
            "payload_ids": payload_ids,
            "page_count": len(payload_ids),
        },
    )
    return copy.deepcopy(payload)


def _response_for_table(response_text: str) -> str:
    text = str(response_text).split("<|im_end|>", 1)[0]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def validate_render_answer(response_text: str, runtime: Any) -> RenderValidation:
    """Validate item/set/list/table shape and payload-reference scope."""

    from rlinf.agents.wideseek_r2.utils.reward import extract_final_answer

    text = _response_for_table(response_text)
    frame = extract_final_answer(text, mode="markdown", strict=False)
    if frame is None:
        return RenderValidation(False, "FORMAT_INVALID", "Expected one Markdown table")
    if frame.empty:
        return RenderValidation(False, "EMPTY_RENDER", "Render table has no rows")
    columns = tuple(str(column).strip() for column in frame.columns)
    expected = runtime.render_payload.get("columns") or runtime.format_requirements.get(
        "columns", []
    )
    expected = tuple(str(column).strip() for column in expected)
    if expected and tuple(column.casefold() for column in columns) != tuple(
        column.casefold() for column in expected
    ):
        return RenderValidation(
            False,
            "COLUMNS_MISMATCH",
            f"Expected columns {expected}, got {columns}",
            len(frame),
            columns,
        )
    if runtime.answer_type == "item" and len(frame) != 1:
        return RenderValidation(
            False, "ITEM_CARDINALITY", "Item output requires one row"
        )
    if runtime.answer_type == "item":
        expected_item_rows = runtime.render_payload.get("rows", ())
        if runtime.config.item_require_terminal_fact and len(expected_item_rows) != 1:
            return RenderValidation(
                False,
                "ITEM_EVIDENCE_CARDINALITY",
                "Item Render Payload must contain exactly one terminal Fact",
                len(frame),
                columns,
            )
        value = str(frame.iat[0, 0]).strip()
        if not value or value.casefold() in {"nan", "none"}:
            return RenderValidation(
                False,
                "EMPTY_ITEM",
                "Item output requires a non-empty value",
                len(frame),
                columns,
            )
    if runtime.answer_type == "set":
        values = [
            tuple(str(value).strip().casefold() for value in row)
            for row in frame.values
        ]
        if len(values) != len(set(values)):
            return RenderValidation(
                False, "SET_DUPLICATE", "Set output contains duplicates"
            )
    explicit_refs = set(
        re.findall(
            r"(?:evidence|entity|source|fact|claim|conflict):[A-Za-z0-9_.:-]+", text
        )
    )
    allowed = set(runtime.render_payload.get("allowed_refs", ()))
    outside = tuple(sorted(ref for ref in explicit_refs if ref not in allowed))
    if outside:
        return RenderValidation(
            False,
            "OUTSIDE_PAYLOAD_REF",
            "Answer references nodes outside the Render Payload",
            len(frame),
            columns,
            outside,
        )
    return RenderValidation(True, row_count=len(frame), columns=columns)


def combine_render_pages(responses: list[str], runtime: Any) -> str:
    """Combine valid physical page tables into one logical Markdown table."""

    from rlinf.agents.wideseek_r2.utils.reward import extract_final_answer

    frames = [
        extract_final_answer(response, mode="markdown", strict=False)
        for response in responses
    ]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        return "\n\n".join(responses)
    columns = [str(column) for column in frames[0].columns]
    rows = [
        [_escape(value) for value in row]
        for frame in frames
        for row in frame.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(_escape(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    del runtime  # The runtime is part of the public helper contract.
    return "```markdown\n" + "\n".join([header, separator, *body]) + "\n```"


def record_render_outcome(runtime: Any, validation: RenderValidation) -> None:
    """Update the Render node and append a bounded outcome record."""

    state = "passed" if validation.valid else "format_retry"
    runtime.render_records.append(
        {
            "attempt": runtime.render_attempt,
            "graph_version": runtime.version,
            "status": state,
            "code": validation.code,
            "message": validation.message,
        }
    )
    render_node = runtime.activation_dag.renders.get(f"render:{runtime.render_attempt}")
    if render_node is not None:
        runtime.activation_dag.replace_node(
            RenderNode(
                **{
                    **render_node.__dict__,
                    "state": state,
                    "outcome": {
                        "code": validation.code,
                        "message": validation.message,
                    },
                }
            )
        )
    if not validation.valid:
        runtime._append_event(
            GraphEventType.FORMAT_RETRY,
            actor_role="system",
            node_ids=(f"render:{runtime.render_attempt}",),
            payload={"code": validation.code, "message": validation.message},
        )


__all__ = [
    "RenderError",
    "RenderValidation",
    "audit_item",
    "build_render_payload",
    "combine_render_pages",
    "record_render_outcome",
    "render",
    "start_render",
    "validate_render_answer",
]
