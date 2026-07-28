# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deterministic Phase 2 retrieval and payload projection."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.embedding_index import (
    DeterministicEmbeddingIndex,
)
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    EvidenceKind,
    EvidenceNode,
    PayloadNode,
)

_SEMANTIC_KINDS = {EvidenceKind.ENTITY, EvidenceKind.FACT}
_SEMANTIC_BODY_KINDS = {
    EvidenceKind.ENTITY,
    EvidenceKind.FACT,
    EvidenceKind.SOURCE,
}
_FOCUS_BODY_KINDS = {
    EvidenceKind.ENTITY,
    EvidenceKind.SOURCE,
    EvidenceKind.CANDIDATE,
    EvidenceKind.CLAIM,
    EvidenceKind.FACT,
    EvidenceKind.CONFLICT,
}
_TRAVERSAL_RELATIONS = {
    "SAME_AS",
    "ABOUT",
    "DERIVED_FROM",
    "VERIFIED_AS",
    "SUPPORTED_BY",
    "SUPPORTS",
    "OBSERVED_IN",
    "NORMALIZES_TO",
    "CONTRADICTS",
    "REFUTES",
    "HAS_FIELD",
    "MEMBER_OF",
}


@dataclass(frozen=True)
class RetrievalCandidate:
    """One node reached from a semantic seed or focus reference."""

    node_id: str
    distance: int
    similarity: float
    seed_ref: str
    focus: bool = False


@dataclass(frozen=True)
class PayloadBuildResult:
    """Materialized action payloads and explicit context failures."""

    payloads: tuple[PayloadNode, ...] = ()
    missing_context: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether all required context was materialized."""

        return not self.missing_context and bool(self.payloads)


def _resolve_ref(graph: Any, ref: str) -> EvidenceNode | None:
    value = str(ref)
    node = graph.nodes.get(value)
    if node is not None and node.active:
        return node
    return graph.get_by_canonical(value, active_only=True)


def _neighbors(graph: Any, node_id: str) -> list[str]:
    """Return the undirected projection neighborhood for one active node."""

    neighbors: set[str] = set()
    for edge in (*graph.incoming(node_id), *graph.outgoing(node_id)):
        if edge.relation not in _TRAVERSAL_RELATIONS:
            continue
        other_id = edge.source_id if edge.target_id == node_id else edge.target_id
        other = graph.nodes.get(other_id)
        if other is not None and other.active:
            neighbors.add(other_id)
    return sorted(neighbors)


def _bfs(
    graph: Any,
    seed: EvidenceNode,
    *,
    similarity: float,
    max_distance: int,
    focus: bool = False,
) -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    queue: deque[tuple[str, int]] = deque([(seed.node_id, 0)])
    seen = {seed.node_id}
    while queue:
        node_id, distance = queue.popleft()
        node = graph.nodes.get(node_id)
        if node is None or not node.active:
            continue
        candidates.append(
            RetrievalCandidate(
                node_id=node_id,
                distance=distance,
                similarity=similarity,
                seed_ref=seed.node_id,
                focus=focus,
            )
        )
        if distance >= max_distance:
            continue
        for neighbor_id in _neighbors(graph, node_id):
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                queue.append((neighbor_id, distance + 1))
    return candidates


def _truncate_excerpt(value: Any, max_tokens: int) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    words = text.split()
    return " ".join(words[: max(0, int(max_tokens))])


def _node_record(
    node: EvidenceNode,
    *,
    runtime: Any,
    include_excerpt: bool,
) -> dict[str, Any]:
    payload = copy.deepcopy(node.payload)
    if node.kind == EvidenceKind.SOURCE:
        # Source content is quoted data.  Preserve provenance metadata while
        # bounding the only potentially large fields.
        for key in ("content", "text", "raw_text", "excerpt"):
            if key in payload:
                if include_excerpt and runtime.config.payload_include_source_excerpt:
                    payload[key] = _truncate_excerpt(
                        payload[key], runtime.config.max_source_excerpt_tokens
                    )
                else:
                    payload.pop(key, None)
    return {
        "node_id": node.node_id,
        "kind": node.kind.value,
        "canonical_key": node.canonical_key,
        "status": node.status.value,
        "active": node.active,
        "payload": payload,
        "provenance": {
            "created_by_action": node.created_by_action,
            "created_by_sub_traj": node.created_by_sub_traj,
            "proposed_by_role": node.proposed_by_role,
            "proposed_by_turn": node.proposed_by_turn,
            "created_at_version": node.created_at_version,
            "updated_at_version": node.updated_at_version,
            "tool_result_refs": list(node.tool_result_refs),
        },
    }


def _estimate_tokens(value: Any) -> int:
    return max(1, len(json.dumps(value, ensure_ascii=False, default=str)) // 4)


def _payload_id(action_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{action_id}|{label}".encode("utf-8")).hexdigest()[:12]
    return f"payload:{action_id}:{digest}"


def materialize_action_payloads(
    runtime: Any,
    *,
    action_id: str,
    subtask: str,
    focus_refs: tuple[str, ...] = (),
    index: DeterministicEmbeddingIndex | None = None,
) -> PayloadBuildResult:
    """Build bounded focus and semantic payloads for one dynamic Action.

    Candidate/Claim/Conflict nodes can guide a focus packet, but only active
    Entity and Fact nodes enter the semantic index.  Every body node has one
    deterministic owner; other packets retain the reference in metadata.
    """

    graph = runtime.active_graph
    embedding_index = index or DeterministicEmbeddingIndex(runtime.config.embedding_dim)
    embedding_index.rebuild(graph)
    query = f"{runtime.question}\nCurrent subtask: {subtask}"
    matches = embedding_index.search(query, runtime.config.payload_top_k)

    focus_nodes: list[EvidenceNode] = []
    missing: list[str] = []
    for ref in focus_refs:
        node = _resolve_ref(graph, str(ref))
        if node is None:
            missing.append(f"unknown_focus_ref:{ref}")
        elif node.node_id not in {item.node_id for item in focus_nodes}:
            focus_nodes.append(node)

    semantic_limit = runtime.config.payload_top_k
    if focus_nodes:
        semantic_limit = max(0, semantic_limit - 1)
    matches = matches[:semantic_limit]

    packets: list[tuple[str, str | None, bool, list[RetrievalCandidate]]] = []
    if focus_nodes:
        focus_candidates: list[RetrievalCandidate] = []
        for node in focus_nodes:
            focus_candidates.extend(
                _bfs(
                    graph,
                    node,
                    similarity=1.0,
                    max_distance=runtime.config.payload_max_distance,
                    focus=True,
                )
            )
        packets.append(("focus", focus_nodes[0].node_id, True, focus_candidates))
    for match in matches:
        node = graph.nodes.get(match.node_id)
        if node is not None:
            packets.append(
                (
                    f"seed:{match.node_id}",
                    match.node_id,
                    False,
                    _bfs(
                        graph,
                        node,
                        similarity=match.similarity,
                        max_distance=runtime.config.payload_max_distance,
                    ),
                )
            )

    if not packets and not missing:
        missing.append("no_active_semantic_seed")
    if missing:
        return PayloadBuildResult(missing_context=tuple(sorted(set(missing))))

    # Candidate owner is focus first, then the exact deterministic rule from
    # the plan: min(distance, -similarity, seed.node_id).
    owners: dict[str, tuple[int, float, str, str]] = {}
    candidate_by_packet: dict[str, list[RetrievalCandidate]] = {}
    for label, seed_ref, is_focus, candidates in packets:
        if seed_ref is None:
            continue
        for candidate in candidates:
            node = graph.nodes.get(candidate.node_id)
            if node is None:
                continue
            if is_focus:
                if node.kind not in _FOCUS_BODY_KINDS:
                    continue
            elif node.kind not in _SEMANTIC_BODY_KINDS:
                continue
            owner_key = (
                0 if candidate.focus else 1,
                candidate.distance,
                -candidate.similarity,
                candidate.seed_ref,
            )
            current = owners.get(candidate.node_id)
            if current is None or owner_key < current:
                owners[candidate.node_id] = (*owner_key, label)
            candidate_by_packet.setdefault(label, []).append(candidate)

    records_by_id: dict[str, dict[str, Any]] = {}
    record_meta: dict[str, tuple[int, int, float, str]] = {}
    focus_root_ids = {node.node_id for node in focus_nodes}
    semantic_seed_ids = {match.node_id for match in matches}
    for node_id, owner in owners.items():
        node = graph.nodes[node_id]
        record = _node_record(
            node,
            runtime=runtime,
            include_excerpt=True,
        )
        records_by_id[node_id] = record
        if node_id in focus_root_ids:
            priority = 0
        elif owner[0] == 0:
            priority = 1
        elif node_id in semantic_seed_ids:
            priority = 1
        elif node.kind == EvidenceKind.FACT:
            priority = 2
        elif node.kind == EvidenceKind.ENTITY:
            priority = 3
        elif node.kind == EvidenceKind.SOURCE:
            priority = 4
        else:
            priority = 5
        record_meta[node_id] = (
            priority,
            owner[1],
            -owner[2],
            owner[3],
        )

    # Required focus and seed nodes are packed before optional neighborhood
    # members.  If a required node cannot fit, the caller receives an explicit
    # MISSING_CONTEXT result rather than a silently incomplete action.
    required_ids: set[str] = set()
    for label, seed_ref, is_focus, candidates in packets:
        if seed_ref is not None:
            required_ids.add(seed_ref)
        if is_focus:
            required_ids.update(
                candidate.node_id for candidate in candidates if candidate.distance == 0
            )
    ordered_ids = sorted(
        records_by_id,
        key=lambda node_id: (
            0 if node_id in required_ids else 1,
            *record_meta[node_id],
            node_id,
        ),
    )
    selected: set[str] = set()
    used_tokens = 0
    payload_budget = max(1, runtime.config.max_payload_tokens - 32)
    for node_id in ordered_ids:
        record_tokens = _estimate_tokens(records_by_id[node_id])
        is_required = node_id in required_ids
        over = (
            len(selected) >= runtime.config.max_payload_nodes
            or used_tokens + record_tokens > payload_budget
        )
        if over and is_required:
            return PayloadBuildResult(
                missing_context=(f"required_payload_budget:{node_id}",),
                metadata={"required_node": node_id},
            )
        if over:
            continue
        selected.add(node_id)
        used_tokens += record_tokens

    payloads: list[PayloadNode] = []
    all_selected = set(selected)
    for label, seed_ref, is_focus, candidates in packets:
        packet_nodes: list[tuple[int, float, str, EvidenceNode]] = []
        refs_only: list[str] = []
        distances: dict[str, int] = {}
        scores: dict[str, float] = {}
        for candidate in candidates:
            node = graph.nodes.get(candidate.node_id)
            if node is None:
                continue
            distances[node.node_id] = min(
                candidate.distance, distances.get(node.node_id, candidate.distance)
            )
            scores[node.node_id] = max(
                candidate.similarity, scores.get(node.node_id, candidate.similarity)
            )
            owner = owners.get(node.node_id, (9, 9, 0.0, "", ""))[-1]
            if owner != label or node.node_id not in all_selected:
                refs_only.append(node.node_id)
                continue
            packet_nodes.append(
                (
                    candidate.distance,
                    -candidate.similarity,
                    node.node_id,
                    node,
                )
            )
        packet_nodes.sort(key=lambda item: item[:3])
        body = [
            _node_record(node, runtime=runtime, include_excerpt=True)
            for _, _, _, node in packet_nodes
        ]
        if not body:
            continue
        payload_metadata = {
            "query": query,
            "semantic_seed": not is_focus,
            "seed_similarity": max(scores.values()) if scores else 1.0,
            "distances": dict(sorted(distances.items())),
            "dedup_refs": sorted(set(refs_only)),
            "truncated": bool(refs_only),
            "untrusted_source_data": True,
        }
        payload_body = {
            "graph_version": runtime.version,
            "nodes": body,
            "refs_only": sorted(set(refs_only)),
        }
        payloads.append(
            PayloadNode(
                payload_id=_payload_id(action_id, label),
                selector={"dynamic_action": action_id},
                projection={
                    "relations": sorted(_TRAVERSAL_RELATIONS),
                    "max_distance": runtime.config.payload_max_distance,
                },
                max_tokens=runtime.config.max_payload_tokens,
                required=True,
                target_node_id=seed_ref,
                seed_ref=seed_ref,
                focus_refs=tuple(node.node_id for node in focus_nodes)
                if is_focus
                else (),
                evidence_refs=tuple(item["node_id"] for item in body),
                graph_version=runtime.version,
                token_count=_estimate_tokens(payload_body),
                retrieval_metadata=payload_metadata,
                body=payload_body,
            )
        )

    if not payloads:
        return PayloadBuildResult(missing_context=("empty_payload",))
    return PayloadBuildResult(
        payloads=tuple(payloads),
        metadata={
            "query": query,
            "seed_count": len(matches),
            "focus_count": len(focus_nodes),
            "selected_nodes": len(selected),
            "token_count": used_tokens,
            "embedding_size": embedding_index.size,
        },
    )


__all__ = ["PayloadBuildResult", "materialize_action_payloads"]
