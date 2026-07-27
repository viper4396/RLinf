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

"""Deterministic render and completion audit for graph-memory MVP."""

from __future__ import annotations

from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.schema import (
    AuditResult,
    EvidenceKind,
    EvidenceStatus,
    GapReport,
    TaskContract,
)


class RenderError(RuntimeError):
    """Raised when deterministic rendering is attempted before audit passes."""


def _terminal_facts(runtime: Any) -> list[Any]:
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
    """Render exactly one accepted Fact as the required fenced Markdown table."""

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
