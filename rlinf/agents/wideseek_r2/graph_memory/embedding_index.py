# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small deterministic embedding adapter used by graph-memory Phase 2.

The rollout path must not depend on a second model service merely to select
context.  The default adapter is therefore a signed hashed-token embedding.
It has the same adapter boundary as a remote embedding index, is reproducible
across processes, and can be replaced later without changing payload or agent
code.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.schema import EvidenceKind, EvidenceNode

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class EmbeddingMatch:
    """One deterministic semantic seed candidate."""

    node_id: str
    similarity: float


def _canonical_name(node: EvidenceNode, graph: Any | None = None) -> str:
    """Return the stable human-readable name used in retrieval text."""

    if node.kind == EvidenceKind.ENTITY:
        return str(
            node.payload.get("canonical_name")
            or node.payload.get("name")
            or node.canonical_key
        )
    if node.kind == EvidenceKind.FACT:
        subject_ref = node.payload.get("subject_ref", "")
        subject = graph.nodes.get(str(subject_ref)) if graph is not None else None
        if subject is not None:
            return str(
                subject.payload.get("canonical_name")
                or subject.payload.get("name")
                or subject.canonical_key
            )
        return str(subject_ref)
    return node.canonical_key


def node_embedding_text(node: EvidenceNode, graph: Any | None = None) -> str:
    """Build the Phase 2 text representation for an active seed node."""

    if node.kind == EvidenceKind.ENTITY:
        aliases = node.payload.get("aliases", [])
        if not isinstance(aliases, (list, tuple)):
            aliases = [aliases]
        return " ".join(
            (
                str(node.payload.get("entity_type", "entity")),
                _canonical_name(node, graph),
                " ".join(str(alias) for alias in aliases),
                str(node.payload.get("description", "")),
            )
        ).strip()
    if node.kind == EvidenceKind.FACT:
        qualifiers = node.payload.get("qualifiers", {})
        if isinstance(qualifiers, dict):
            qualifier_text = " ".join(
                f"{key} {value}" for key, value in sorted(qualifiers.items())
            )
        else:
            qualifier_text = str(qualifiers)
        return " ".join(
            (
                _canonical_name(node, graph),
                str(node.payload.get("predicate", "")),
                str(node.payload.get("object", node.payload.get("value", ""))),
                qualifier_text,
            )
        ).strip()
    return " ".join(
        [node.canonical_key]
        + [str(value) for value in node.payload.values() if value is not None]
    ).strip()


class DeterministicEmbeddingIndex:
    """A normalized hashed-token index with deterministic cosine search."""

    def __init__(self, dimension: int = 128):
        self.dimension = max(8, int(dimension))
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._texts: dict[str, str] = {}

    def _vector(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(str(text).casefold())
        # Include adjacent pairs so short entity names and predicates retain
        # a little word-order information without requiring a tokenizer.
        tokens.extend(f"{left}:{right}" for left, right in zip(tokens, tokens[1:]))
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return tuple(values)
        return tuple(value / norm for value in values)

    def clear(self) -> None:
        """Remove all indexed nodes."""

        self._vectors.clear()
        self._texts.clear()

    def upsert(self, node_id: str, text: str) -> None:
        """Insert or replace one indexed node."""

        node_id = str(node_id)
        self._texts[node_id] = str(text)
        self._vectors[node_id] = self._vector(text)

    def rebuild(self, graph: Any) -> None:
        """Index only active Entity and Fact nodes from ``graph``."""

        self.clear()
        for node in sorted(graph.nodes.values(), key=lambda item: item.node_id):
            if node.active and node.kind in {EvidenceKind.ENTITY, EvidenceKind.FACT}:
                self.upsert(node.node_id, node_embedding_text(node, graph))

    def search(self, query: str, top_k: int) -> list[EmbeddingMatch]:
        """Return deterministic similarity-ranked semantic seeds."""

        query_vector = self._vector(query)
        ranked: list[EmbeddingMatch] = []
        for node_id, vector in self._vectors.items():
            similarity = sum(left * right for left, right in zip(query_vector, vector))
            ranked.append(EmbeddingMatch(node_id=node_id, similarity=float(similarity)))
        ranked.sort(key=lambda item: (-item.similarity, item.node_id))
        return ranked[: max(0, int(top_k))]

    @property
    def size(self) -> int:
        """Number of active semantic seed nodes."""

        return len(self._vectors)


__all__ = [
    "DeterministicEmbeddingIndex",
    "EmbeddingMatch",
    "node_embedding_text",
]
