# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Metric-only graph lineage attribution for Phase 1."""

from __future__ import annotations

from collections import Counter
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.schema import EvidenceKind, EvidenceStatus


def graph_credit_metrics(runtime: Any) -> dict[str, Any]:
    """Return accepted/consumed lineage metrics without changing reward."""

    graph = runtime.evidence_graph
    accepted = [
        node
        for node in graph.nodes.values()
        if node.kind in {EvidenceKind.CLAIM, EvidenceKind.FACT}
        and node.status == EvidenceStatus.VERIFIED
    ]
    consumed = set()
    audit = getattr(runtime, "last_audit", None)
    if audit is not None:
        consumed.update(audit.fact_refs)
    by_subtraj = Counter(node.created_by_sub_traj for node in accepted)
    return {
        "accepted_evidence": len(accepted),
        "consumed_fact_count": len(consumed),
        "accepted_by_sub_traj": dict(by_subtraj),
        "open_conflict_count": sum(
            node.kind == EvidenceKind.CONFLICT
            and node.status.value in {"open", "conflicted"}
            for node in graph.nodes.values()
        ),
    }
