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

"""Scoped evidence selection and token-budget enforcement."""

from __future__ import annotations

import copy
import json
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.schema import (
    EvidenceKind,
    EvidenceNode,
    PayloadNode,
)


class EvidenceAccessError(PermissionError):
    """Raised when a worker reads outside its activation scope."""


def estimate_tokens(value: Any) -> int:
    """Conservatively estimate tokens for a JSON-like payload."""

    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, (len(value) + 3) // 4)
    return max(1, (len(json.dumps(value, ensure_ascii=False, default=str)) + 3) // 4)


def _project_node(
    node: EvidenceNode, projection: dict[str, Any] | None = None
) -> dict[str, Any]:
    projection = projection or {}
    payload = copy.deepcopy(node.payload)
    if (
        not projection.get("include_source_excerpt", True)
        and node.kind == EvidenceKind.SOURCE
    ):
        for key in ("excerpt", "content", "text", "body"):
            payload.pop(key, None)
    fields = projection.get("fields")
    if isinstance(fields, (list, tuple)):
        payload = {key: payload.get(key) for key in fields if key in payload}
    return {
        "ref": node.node_id,
        "kind": node.kind.value,
        "status": node.status.value,
        "canonical_key": node.canonical_key,
        "payload": payload,
        "tags": list(node.tags),
        "created_by_action": node.created_by_action,
    }


def _source_refs(runtime: Any, node_id: str) -> set[str]:
    graph = runtime.evidence_graph
    refs = set()
    pending = [node_id]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for edge in graph.incoming(current):
            source = graph.nodes.get(edge.source_id)
            if source and source.kind in {EvidenceKind.SOURCE, EvidenceKind.ENTITY}:
                refs.add(source.node_id)
            elif source and edge.relation in {
                "VERIFIED_AS",
                "DERIVED_FROM",
                "ABOUT",
                "SUPPORTS",
                "NORMALIZES_TO",
            }:
                pending.append(source.node_id)
        for edge in graph.outgoing(current, "OBSERVED_IN"):
            source = graph.nodes.get(edge.target_id)
            if source and source.kind == EvidenceKind.SOURCE:
                refs.add(source.node_id)
        for edge in graph.outgoing(current):
            if edge.relation not in {"ABOUT", "MEMBER_OF"}:
                continue
            related = graph.nodes.get(edge.target_id)
            if related and related.kind == EvidenceKind.ENTITY:
                refs.add(related.node_id)
    return refs


def select_evidence(
    runtime: Any,
    refs: list[str] | tuple[str, ...] | None = None,
    *,
    allowed_reads: set[str] | frozenset[str] | None = None,
    projection: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    since_version: int | None = None,
) -> list[dict[str, Any]]:
    """Select only explicitly scoped evidence references.

    ``allowed_reads`` is mandatory for worker reads.  System callers may pass
    ``None`` while materializing a payload, but even then only requested refs
    (or selector-derived refs) are returned.
    """

    graph = runtime.evidence_graph
    requested = list(refs or [])
    if not requested:
        return []
    resolved_refs = []
    seen_refs: set[str] = set()
    for ref in requested:
        node = graph.nodes.get(ref) or graph.get_by_canonical(ref)
        if node is None:
            raise KeyError(f"Unknown evidence ref: {ref}")
        if node.node_id not in seen_refs:
            resolved_refs.append(node.node_id)
            seen_refs.add(node.node_id)
    if allowed_reads is not None:
        forbidden = set(resolved_refs) - set(allowed_reads)
        if forbidden:
            raise EvidenceAccessError(
                f"Evidence refs are outside the action scope: {sorted(forbidden)}"
            )
    result: list[dict[str, Any]] = []
    token_count = 0
    for ref in resolved_refs:
        node = graph.nodes[ref]
        if node.kind == EvidenceKind.SOURCE:
            uri = node.payload.get("uri")
            locator = node.payload.get("locator")
            if (
                not isinstance(uri, str)
                or not uri.strip()
                or locator is None
                or not str(locator).strip()
            ):
                raise EvidenceAccessError(
                    f"Source {node.node_id!r} is missing uri/locator provenance"
                )
        if since_version is not None and node.created_at_version <= since_version:
            continue
        projected = _project_node(node, projection)
        projected_tokens = estimate_tokens(projected)
        if max_tokens is not None and token_count + projected_tokens > max_tokens:
            break
        result.append(projected)
        token_count += projected_tokens
    return result


def _selector_refs(runtime: Any, payload: PayloadNode) -> list[str]:
    selector = payload.selector or {}
    explicit = selector.get("refs")
    if isinstance(explicit, str):
        explicit = [explicit]
    if explicit:
        return [str(ref) for ref in explicit]

    refs_from_gate = selector.get("refs_from_gate")
    if refs_from_gate:
        gate_id = str(refs_from_gate)
        refs = set()
        gate = runtime.activation_dag.gates.get(gate_id)
        if gate is not None:
            condition = gate.condition
            if condition.get("op") == "fact_exists":
                fact = runtime.evidence_graph.get_by_canonical(
                    condition.get("canonical_key", ""), EvidenceKind.FACT
                )
                if fact is None and condition.get("ref"):
                    fact = runtime.evidence_graph.nodes.get(
                        condition["ref"]
                    ) or runtime.evidence_graph.get_by_canonical(condition["ref"])
                if fact is not None:
                    refs.add(fact.node_id)
                    refs.update(_source_refs(runtime, fact.node_id))
            for child in condition.get("conditions", []):
                child_ref = child.get("ref") or child.get("canonical_key")
                if child_ref:
                    node = runtime.evidence_graph.nodes.get(
                        child_ref
                    ) or runtime.evidence_graph.get_by_canonical(child_ref)
                    if node is not None:
                        refs.add(node.node_id)
                        refs.update(_source_refs(runtime, node.node_id))
        refs.update(runtime.action_acl.get(gate_id, set()))
        return sorted(refs)

    bound_entity = selector.get("entity_ref")
    if bound_entity:
        entity = runtime.evidence_graph.nodes.get(
            str(bound_entity)
        ) or runtime.evidence_graph.get_by_canonical(
            str(bound_entity), EvidenceKind.ENTITY
        )
        bound_id = entity.node_id if entity is not None else str(bound_entity)
        refs = [bound_id]
        if selector.get("include_about_entities", True):
            graph = runtime.evidence_graph
            for edge in graph.edges.values():
                if edge.target_id == bound_id or edge.source_id == bound_id:
                    refs.extend([edge.source_id, edge.target_id])
        if selector.get("include_source_refs", True):
            refs.extend(_source_refs(runtime, bound_id))
        return list(dict.fromkeys(refs))

    if selector.get("task_context"):
        return []
    if selector.get("all_ready_evidence"):
        return [
            node.node_id
            for node in runtime.evidence_graph.nodes.values()
            if node.kind in {EvidenceKind.ENTITY, EvidenceKind.FACT}
        ]
    return []


def materialize_payload(
    runtime: Any,
    payload: PayloadNode,
    *,
    action_id: str,
) -> dict[str, Any] | None:
    """Materialize one payload recipe into a bounded activation packet."""

    selector = payload.selector or {}
    if selector.get("task_context"):
        packet = {
            "payload_id": payload.payload_id,
            "evidence": [],
            "allowed_reads": [],
            "task_context": runtime.task_context(),
        }
        runtime.action_acl.setdefault(action_id, set())
        return packet
    refs = _selector_refs(runtime, payload)
    if payload.required and not refs:
        return None
    selected = select_evidence(
        runtime,
        refs,
        projection=payload.projection,
        max_tokens=payload.max_tokens,
    )
    allowed = {item["ref"] for item in selected}
    runtime.action_acl.setdefault(action_id, set()).update(allowed)
    return {
        "payload_id": payload.payload_id,
        "evidence": selected,
        "allowed_reads": sorted(allowed),
    }


def read_scoped_evidence(
    runtime: Any,
    *,
    action_id: str,
    refs: list[str] | tuple[str, ...],
    fields: list[str] | tuple[str, ...] | None = None,
    since_version: int | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Read refs permitted by the action ACL."""

    if action_id not in runtime.activation_dag.actions:
        raise EvidenceAccessError(f"Unknown action scope {action_id!r}")
    expiry = runtime.action_acl_expiry.get(action_id)
    if expiry is not None and runtime.version > expiry:
        raise EvidenceAccessError(
            f"Activation scope for action {action_id!r} expired at graph version {expiry}"
        )
    allowed = runtime.action_acl.get(action_id, set())
    projection: dict[str, Any] = {}
    if fields:
        projection["fields"] = list(fields)
    token_budget = runtime.config.max_selected_evidence_tokens
    if max_tokens is not None:
        token_budget = min(token_budget, max(0, int(max_tokens)))
    return select_evidence(
        runtime,
        refs,
        allowed_reads=allowed,
        projection=projection,
        max_tokens=token_budget,
        since_version=since_version,
    )
