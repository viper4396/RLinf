# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Phase 4 completion audit and workflow state transitions."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.item import check_item_completion
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionState,
    AuditNode,
    EvidenceKind,
    EvidenceStatus,
    GraphEventType,
    PayloadNode,
)

_TERMINAL_ACTION_STATES = {
    ActionState.COMPLETED,
    ActionState.FAILED,
    ActionState.BLOCKED,
    ActionState.INVALIDATED,
    ActionState.CONSUMED,
}
_INVALID_FACT_STATES = {
    EvidenceStatus.DISPUTED,
    EvidenceStatus.INVALIDATED,
    EvidenceStatus.REFUTED,
    EvidenceStatus.CONFLICTED,
    EvidenceStatus.REJECTED,
    EvidenceStatus.RETIRED,
}


@dataclass(frozen=True)
class AuditReport:
    """Mechanical result of one Audit attempt."""

    passed: bool
    status: str
    attempt: int
    graph_version: int
    missing: tuple[str, ...] = ()
    invariants: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


def _node_ref(node: Any) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "kind": node.kind.value,
        "canonical_key": node.canonical_key,
        "status": node.status.value,
        "active": node.active,
        "provenance": {
            "created_by_action": node.created_by_action,
            "created_by_sub_traj": node.created_by_sub_traj,
            "proposed_by_role": node.proposed_by_role,
            "proposed_by_turn": node.proposed_by_turn,
            "tool_result_refs": list(node.tool_result_refs),
        },
    }


def _source_refs(runtime: Any, node_id: str) -> list[str]:
    graph = runtime.evidence_graph
    found: set[str] = set()
    queue = [node_id]
    seen: set[str] = set()
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        node = graph.nodes.get(current)
        if node is None or not node.active:
            continue
        if node.kind == EvidenceKind.SOURCE:
            found.add(node.node_id)
            continue
        for ref in node.payload.get("source_refs", ()):
            source = graph.nodes.get(str(ref)) or graph.get_by_canonical(str(ref))
            if source is not None and source.active:
                queue.append(source.node_id)
        for edge in (*graph.incoming(current), *graph.outgoing(current)):
            if edge.relation in {
                "SUPPORTED_BY",
                "OBSERVED_IN",
                "SUPPORTS",
                "VERIFIED_AS",
            }:
                queue.append(
                    edge.source_id if edge.target_id == current else edge.target_id
                )
    return sorted(found)


def terminal_invariants(runtime: Any) -> tuple[bool, dict[str, Any]]:
    """Check the non-negotiable terminal graph conditions."""

    graph = runtime.evidence_graph
    active_claims = graph.iter_kind(EvidenceKind.CLAIM)
    active_facts = graph.iter_kind(EvidenceKind.FACT)
    active_conflicts = [
        node for node in graph.iter_kind(EvidenceKind.CONFLICT) if node.active
    ]
    claims_without_facts = [
        claim.node_id
        for claim in active_claims
        if not graph.facts_for_claim(claim.node_id)
    ]
    facts_without_claims = []
    facts_without_sources = []
    invalid_facts = []
    for fact in active_facts:
        claim_refs = [
            edge.source_id for edge in graph.incoming(fact.node_id, "VERIFIED_AS")
        ]
        claim_ref = fact.payload.get("claim_ref")
        if claim_ref:
            claim = graph.nodes.get(str(claim_ref))
            if claim is not None and claim.active and claim.kind == EvidenceKind.CLAIM:
                claim_refs.append(claim.node_id)
        if not any(
            ref in graph.nodes
            and graph.nodes[ref].active
            and graph.nodes[ref].kind == EvidenceKind.CLAIM
            for ref in claim_refs
        ):
            facts_without_claims.append(fact.node_id)
        if not _source_refs(runtime, fact.node_id):
            facts_without_sources.append(fact.node_id)
        if (
            fact.status in _INVALID_FACT_STATES
            or fact.status != EvidenceStatus.VERIFIED
        ):
            invalid_facts.append(fact.node_id)
    nonterminal_actions = [
        action_id
        for action_id, action in sorted(runtime.activation_dag.actions.items())
        if action.state not in _TERMINAL_ACTION_STATES
    ]
    invariants = {
        "no_pending_claims": not runtime.pending_claim_ids,
        "no_pending_conflicts": not runtime.pending_conflict_ids,
        "no_active_claim_without_fact": not claims_without_facts,
        "no_active_conflict": not active_conflicts,
        "every_fact_has_active_claim": not facts_without_claims,
        "every_fact_has_source": not facts_without_sources,
        "no_invalidated_fact": not invalid_facts,
        "all_actions_terminal": not nonterminal_actions,
        "no_pending_memory_transaction": not runtime.pending_memory_transaction,
        "details": {
            "claims_without_facts": claims_without_facts,
            "active_conflicts": [node.node_id for node in active_conflicts],
            "facts_without_claims": facts_without_claims,
            "facts_without_sources": facts_without_sources,
            "invalid_facts": invalid_facts,
            "nonterminal_actions": nonterminal_actions,
        },
    }
    if runtime.answer_type == "item" and runtime.config.item_require_terminal_fact:
        item_completion = check_item_completion(runtime)
        invariants.update(
            {
                "item_has_terminal_fact": "item_terminal_fact"
                not in item_completion.missing,
                "item_unique_terminal_fact": "item_unique_terminal_fact"
                not in item_completion.missing,
                "item_terminal_value_present": "item_terminal_value"
                not in item_completion.missing,
            }
        )
        invariants["details"]["item_terminal_fact_refs"] = list(
            item_completion.terminal_refs
        )
        invariants["details"]["item_terminal_values"] = list(item_completion.values)
        invariants["details"]["item_missing"] = list(item_completion.missing)
    passed = all(value for key, value in invariants.items() if key != "details")
    return passed, invariants


def build_audit_payload(runtime: Any, response_text: str = "") -> dict[str, Any]:
    """Build the deterministic Main-only Audit context packet."""

    graph = runtime.evidence_graph
    active_claims = graph.iter_kind(EvidenceKind.CLAIM)
    active_conflicts = graph.iter_kind(EvidenceKind.CONFLICT)
    active_entities = graph.iter_kind(EvidenceKind.ENTITY)
    active_facts = graph.iter_kind(EvidenceKind.FACT)
    covered = []
    for fact in active_facts:
        covered.append(
            {
                "fact": _node_ref(fact),
                "claim_refs": [
                    edge.source_id
                    for edge in graph.incoming(fact.node_id, "VERIFIED_AS")
                ],
                "source_refs": _source_refs(runtime, fact.node_id),
            }
        )
    payload = {
        "event": "AUDIT_REQUIRED",
        "graph_version": runtime.version,
        "question": runtime.question,
        "answer_type": runtime.answer_type,
        "format_requirements": copy.deepcopy(runtime.format_requirements),
        "original_response_present": bool(response_text.strip()),
        "pending_claims": sorted(runtime.pending_claim_ids),
        "pending_conflicts": sorted(runtime.pending_conflict_ids),
        "active_claims": [_node_ref(node) for node in active_claims],
        "active_conflicts": [
            {
                **_node_ref(node),
                "competing_refs": list(node.payload.get("competing_refs", ())),
            }
            for node in active_conflicts
        ],
        "completed_actions": [
            {
                "action_id": action_id,
                "state": action.state.value,
                "objective": action.objective,
                "result": copy.deepcopy(runtime.action_results.get(action_id, {})),
            }
            for action_id, action in sorted(runtime.activation_dag.actions.items())
            if action.state in _TERMINAL_ACTION_STATES
        ],
        "active_entity_refs": [node.node_id for node in active_entities],
        "active_fact_coverage": covered,
        "pending_memory_transaction": runtime.pending_memory_transaction,
    }
    if runtime.answer_type == "item":
        item_completion = check_item_completion(runtime)
        payload["item_completion"] = {
            "required": runtime.config.item_require_terminal_fact,
            "terminal_fact_refs": list(item_completion.terminal_refs),
            "values": list(item_completion.values),
            "missing": list(item_completion.missing),
            "instruction": (
                "The final item must be exactly one non-empty value from the "
                "source-backed verified terminal Fact."
            ),
        }
    return payload


def start_audit(runtime: Any, response_text: str = "") -> dict[str, Any]:
    """Create a fresh Audit activation node and enter the Audit phase."""

    runtime.audit_attempt += 1
    runtime.workflow_phase = "audit"
    runtime.audit_payload = build_audit_payload(runtime, response_text)
    payload_id = f"payload:audit:{runtime.audit_attempt}"
    audit_id = f"audit:{runtime.audit_attempt}"
    payload = PayloadNode(
        payload_id=payload_id,
        selector={"audit": True},
        projection={"active_graph": True},
        max_tokens=runtime.config.max_payload_tokens,
        required=True,
        graph_version=runtime.version,
        token_count=max(
            1, len(json.dumps(runtime.audit_payload, ensure_ascii=False)) // 4
        ),
        retrieval_metadata={"phase": "audit", "attempt": runtime.audit_attempt},
        body=copy.deepcopy(runtime.audit_payload),
    )
    runtime.activation_dag.add_node(payload)
    runtime.activation_dag.add_node(
        AuditNode(
            audit_id=audit_id,
            policy={"max_attempts": runtime.config.max_audit_attempts},
            state="required",
            attempt=runtime.audit_attempt,
            graph_version=runtime.version,
            covered_action_seq=runtime.action_sequence,
            payload_ids=(payload_id,),
        )
    )
    runtime._append_event(
        GraphEventType.CREATE_AUDIT,
        actor_role="system",
        node_ids=(audit_id, payload_id),
        payload={
            "attempt": runtime.audit_attempt,
            "graph_version": runtime.version,
            "covered_action_seq": runtime.action_sequence,
        },
    )
    return copy.deepcopy(runtime.audit_payload)


def record_audit_outcome(
    runtime: Any,
    *,
    model_pass: bool,
    response_text: str,
) -> AuditReport:
    """Record model intent plus mechanical invariant outcome."""

    passed, invariants = terminal_invariants(runtime)
    missing = tuple(
        key for key, value in invariants.items() if key != "details" and not value
    )
    if not model_pass:
        missing = tuple(dict.fromkeys((*missing, "audit_pass_marker")))
    status = "PASS" if model_pass and passed else "INCOMPLETE"
    report = AuditReport(
        passed=status == "PASS",
        status=status,
        attempt=runtime.audit_attempt,
        graph_version=runtime.version,
        missing=missing,
        invariants=invariants,
        payload=build_audit_payload(runtime, response_text),
    )
    runtime.last_audit = report
    runtime.audit_records.append(
        {
            "attempt": report.attempt,
            "graph_version": report.graph_version,
            "status": report.status,
            "model_pass": model_pass,
            "missing": list(report.missing),
            "invariants": copy.deepcopy(report.invariants),
        }
    )
    audit_node = runtime.activation_dag.audits.get(f"audit:{runtime.audit_attempt}")
    if audit_node is not None:
        runtime.activation_dag.replace_node(
            AuditNode(
                **{
                    **audit_node.__dict__,
                    "state": "passed" if report.passed else "incomplete",
                    "outcome": {
                        "status": report.status,
                        "missing": list(report.missing),
                    },
                }
            )
        )
    runtime._append_event(
        GraphEventType.AUDIT_OUTCOME,
        actor_role="system",
        node_ids=(f"audit:{runtime.audit_attempt}",),
        payload={
            "attempt": runtime.audit_attempt,
            "status": report.status,
            "missing": list(report.missing),
        },
    )
    return report


def _clean_response(response_text: str) -> str:
    return str(response_text).split("<|im_end|>", 1)[0].strip()


def parse_audit_pass(response_text: str) -> bool:
    """Accept only the structured, no-tool ``AUDIT_PASS`` marker."""

    text = _clean_response(response_text)
    if text == "AUDIT_PASS":
        return True
    candidate = text
    if "```" in candidate:
        candidate = candidate.replace("```json", "").replace("```", "").strip()
    try:
        value = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        match = re.fullmatch(r"\s*\{.*\}\s*", candidate, flags=re.DOTALL)
        if not match:
            return False
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return False
    return (
        isinstance(value, dict)
        and str(value.get("status", value.get("event", ""))).upper() == "AUDIT_PASS"
    )


def audit_feedback(report: AuditReport, *, code: str | None = None) -> str:
    """Return bounded structured feedback for a recoverable Audit failure."""

    return json.dumps(
        {
            "status": "AUDIT_INCOMPLETE",
            "code": code or "AUDIT_REJECTED",
            "attempt": report.attempt,
            "graph_version": report.graph_version,
            "missing": list(report.missing),
            "details": report.invariants.get("details", {}),
            "instruction": "Use exactly one legal graph tool this turn to repair the gap, or output AUDIT_PASS only after all invariants hold.",
        },
        ensure_ascii=False,
    )


__all__ = [
    "AuditReport",
    "audit_feedback",
    "build_audit_payload",
    "parse_audit_pass",
    "record_audit_outcome",
    "start_audit",
    "terminal_invariants",
]
