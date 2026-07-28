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

"""Trajectory-local graph-memory implementation for WideSeek-R2."""

from rlinf.agents.wideseek_r2.graph_memory.audit import AuditReport
from rlinf.agents.wideseek_r2.graph_memory.embedding_index import (
    DeterministicEmbeddingIndex,
    EmbeddingMatch,
)
from rlinf.agents.wideseek_r2.graph_memory.payload_builder import PayloadBuildResult
from rlinf.agents.wideseek_r2.graph_memory.renderer import RenderValidation
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionNode,
    ActionState,
    EvidenceKind,
    EvidenceNode,
    EvidenceProposal,
    GraphConfig,
    GraphEvent,
    GraphEventType,
    NodeProposal,
    TaskContract,
    ToolResultRecord,
)
from rlinf.agents.wideseek_r2.graph_memory.state import (
    ActivationDAG,
    EvidenceGraph,
    GraphRuntime,
)

__all__ = [
    "ActionNode",
    "ActionState",
    "ActivationDAG",
    "EvidenceKind",
    "EvidenceNode",
    "EvidenceProposal",
    "GraphEvent",
    "GraphEventType",
    "EvidenceGraph",
    "GraphConfig",
    "GraphRuntime",
    "NodeProposal",
    "TaskContract",
    "ToolResultRecord",
    "AuditReport",
    "DeterministicEmbeddingIndex",
    "EmbeddingMatch",
    "PayloadBuildResult",
    "RenderValidation",
]
