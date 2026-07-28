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

"""Schemas shared by the WideSeek-R2 graph-memory runtime.

The graph runtime deliberately keeps its public data model independent from
OmegaConf, Ray, and the agent-loop implementation.  This makes proposals easy
to validate in isolation and keeps the canonical state serializable for tests
and bounded rollout snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

JsonValue = Any
AnswerKind = Literal["item", "set", "list", "table"]


class EvidenceKind(str, Enum):
    """Kinds of nodes stored in the Evidence Graph."""

    ENTITY = "entity"
    SOURCE = "source"
    CANDIDATE = "candidate"
    CLAIM = "claim"
    FACT = "fact"
    CONFLICT = "conflict"

    @classmethod
    def coerce(cls, value: str | "EvidenceKind") -> "EvidenceKind":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported evidence kind: {value!r}") from exc


class EvidenceStatus(str, Enum):
    """Lifecycle states for evidence nodes."""

    PROPOSED = "proposed"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONFLICTED = "conflicted"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INVALIDATED = "invalidated"
    OPEN = "open"
    RESOLVED = "resolved"
    ACTIVE = "active"
    PENDING = "pending"
    VERIFYING = "verifying"
    DISPUTED = "disputed"
    INVESTIGATING = "investigating"
    PROMOTED = "promoted"
    RETIRED = "retired"
    MERGED = "merged"


class ActionState(str, Enum):
    """Scheduler state for an activation action."""

    DORMANT = "dormant"
    MATERIALIZING_PAYLOAD = "materializing_payload"
    READY = "ready"
    MISSING_CONTEXT = "missing_context"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    INVALIDATED = "invalidated"
    CONSUMED = "consumed"


class ActivationNodeKind(str, Enum):
    """Kinds of nodes in the Activation DAG."""

    ACTION = "action"
    GATE = "gate"
    PAYLOAD = "payload"
    JOIN = "join"
    AUDIT = "audit"
    RENDER = "render"


class ActivationEdgeType(str, Enum):
    """Allowed control/data edges in the Activation DAG."""

    PRECEDES = "precedes"
    GUARDS = "guards"
    DELIVERS = "delivers"
    JOINS_INTO = "joins_into"
    EXPANDS_TO = "expands_to"
    ON_OUTCOME = "on_outcome"


class DeliveryPolicy(str, Enum):
    """When a payload may be delivered to an action."""

    ON_READY = "on_ready"
    ON_CHANGE = "on_change"
    MANUAL_PULL = "manual_pull"


class CompletenessType(str, Enum):
    """Built-in completion policies supported by Phase 1."""

    DEPENDENCY_TERMINAL = "dependency_terminal"
    CLOSED_SOURCE_EXHAUSTED = "closed_source_exhausted"
    PARTITION_COVERED = "partition_covered"
    FIXED_CARDINALITY = "fixed_cardinality"
    BUDGET_BOUNDED_BEST_EFFORT = "budget_bounded_best_effort"


def _tuple(value: Any) -> tuple:
    """Convert optional JSON lists to tuples without changing strings."""

    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class CompletenessPolicy:
    """Mechanical completion policy for a task contract."""

    type: str = CompletenessType.DEPENDENCY_TERMINAL.value
    partitions: tuple[str, ...] = ()
    expected_count: int | None = None
    allow_best_effort: bool = False

    @classmethod
    def from_dict(cls, value: Any) -> "CompletenessPolicy":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            type=str(
                value.get("type", CompletenessType.DEPENDENCY_TERMINAL.value)
            ).lower(),
            partitions=tuple(str(item) for item in _tuple(value.get("partitions"))),
            expected_count=value.get("expected_count"),
            allow_best_effort=bool(value.get("allow_best_effort", False)),
        )


@dataclass(frozen=True)
class CitationPolicy:
    """Source/provenance requirements for accepted facts."""

    require_source: bool = True
    min_independent_sources: int = 1

    @classmethod
    def from_dict(cls, value: Any) -> "CitationPolicy":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            require_source=bool(value.get("require_source", True)),
            min_independent_sources=max(
                0, int(value.get("min_independent_sources", 1))
            ),
        )


@dataclass(frozen=True)
class OrderSpec:
    """Ordering metadata retained for later list support."""

    field: str
    direction: str = "asc"
    tie_break: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Any) -> "OrderSpec":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            field=str(value.get("field", "")),
            direction=str(value.get("direction", "asc")),
            tie_break=tuple(str(item) for item in _tuple(value.get("tie_break"))),
        )


@dataclass(frozen=True)
class TaskContract:
    """Immutable task/output contract compiled from a model proposal."""

    contract_id: str
    question: str
    answer_kind: AnswerKind
    version: int = 1
    output_columns: tuple[str, ...] = ("Item",)
    row_key: tuple[str, ...] = ()
    membership_rule: str | None = None
    ordering: tuple[OrderSpec, ...] = ()
    completeness_policy: CompletenessPolicy = field(default_factory=CompletenessPolicy)
    citation_policy: CitationPolicy = field(default_factory=CitationPolicy)
    language: str = "en"
    terminal_fact_ref: str | None = None
    terminal_predicate: str | None = None
    final_value_field: str = "value"

    def __post_init__(self) -> None:
        if self.answer_kind not in {"item", "set", "list", "table"}:
            raise ValueError(f"Unsupported answer kind: {self.answer_kind!r}")
        if not self.question:
            raise ValueError("TaskContract.question must not be empty")
        if self.version < 1:
            raise ValueError("TaskContract.version must be positive")

    @classmethod
    def from_dict(cls, value: Any, *, question: str | None = None) -> "TaskContract":
        if isinstance(value, cls):
            return value
        value = value or {}
        answer_kind = str(
            value.get("answer_kind", value.get("answer_type", "item"))
        ).lower()
        columns = value.get("output_columns", value.get("columns", ("Item",)))
        return cls(
            contract_id=str(value.get("contract_id", f"contract:{uuid4().hex[:12]}")),
            question=str(value.get("question", question or "")),
            answer_kind=answer_kind,  # type: ignore[arg-type]
            version=int(value.get("version", 1)),
            output_columns=tuple(str(item) for item in _tuple(columns)) or ("Item",),
            row_key=tuple(str(item) for item in _tuple(value.get("row_key"))),
            membership_rule=value.get("membership_rule"),
            ordering=tuple(
                OrderSpec.from_dict(item) for item in _tuple(value.get("ordering"))
            ),
            completeness_policy=CompletenessPolicy.from_dict(
                value.get("completeness_policy")
            ),
            citation_policy=CitationPolicy.from_dict(value.get("citation_policy")),
            language=str(value.get("language", "en")),
            terminal_fact_ref=value.get("terminal_fact_ref"),
            terminal_predicate=value.get("terminal_predicate"),
            final_value_field=str(value.get("final_value_field", "value")),
        )


@dataclass(frozen=True)
class EvidenceNode:
    """Canonical node in the Evidence Graph."""

    node_id: str
    kind: EvidenceKind
    canonical_key: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    status: EvidenceStatus = EvidenceStatus.PROPOSED
    created_by_action: str = "system"
    created_by_sub_traj: int = 0
    created_at_version: int = 0
    confidence: float | None = None
    tags: tuple[str, ...] = ()
    active: bool = True
    proposed_by_role: Literal["main", "subagent", "system"] = "system"
    proposed_by_turn: int = 0
    updated_at_version: int = 0
    tool_result_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceKind.coerce(self.kind))
        if not isinstance(self.status, EvidenceStatus):
            object.__setattr__(self, "status", EvidenceStatus(str(self.status).lower()))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "tool_result_refs", tuple(self.tool_result_refs))
        if self.proposed_by_role not in {"main", "subagent", "system"}:
            raise ValueError(
                f"Unsupported graph proposer role: {self.proposed_by_role!r}"
            )
        object.__setattr__(self, "active", bool(self.active))

    def with_status(
        self, status: EvidenceStatus, *, version: int | None = None
    ) -> "EvidenceNode":
        return EvidenceNode(
            node_id=self.node_id,
            kind=self.kind,
            canonical_key=self.canonical_key,
            payload=dict(self.payload),
            status=status,
            created_by_action=self.created_by_action,
            created_by_sub_traj=self.created_by_sub_traj,
            created_at_version=self.created_at_version if version is None else version,
            confidence=self.confidence,
            tags=self.tags,
            active=status
            not in {
                EvidenceStatus.RETIRED,
                EvidenceStatus.REJECTED,
                EvidenceStatus.INVALIDATED,
                EvidenceStatus.MERGED,
            },
            proposed_by_role=self.proposed_by_role,
            proposed_by_turn=self.proposed_by_turn,
            updated_at_version=self.updated_at_version if version is None else version,
            tool_result_refs=self.tool_result_refs,
        )


@dataclass(frozen=True)
class EvidenceEdge:
    """Typed relationship in the Evidence Graph."""

    edge_id: str
    source_id: str
    relation: str
    target_id: str
    created_by_action: str = "system"
    created_at_version: int = 0
    created_by_role: Literal["main", "subagent", "system"] = "system"
    proposed_by_turn: int = 0
    active: bool = True


@dataclass(frozen=True)
class NodeProposal:
    """Client-side proposal for one Evidence Graph node."""

    client_ref: str
    kind: EvidenceKind | str
    canonical_key: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    status: EvidenceStatus | str = EvidenceStatus.PROPOSED
    confidence: float | None = None
    tags: tuple[str, ...] = ()
    tool_result_refs: tuple[str, ...] = ()

    def normalized_kind(self) -> EvidenceKind:
        return EvidenceKind.coerce(self.kind)

    def normalized_status(self) -> EvidenceStatus:
        if isinstance(self.status, EvidenceStatus):
            return self.status
        return EvidenceStatus(str(self.status).lower())


@dataclass(frozen=True)
class EdgeProposal:
    """Client-side proposal for one Evidence Graph edge."""

    source_ref: str
    relation: str
    target_ref: str

    @property
    def from_ref(self) -> str:
        return self.source_ref

    @property
    def to_ref(self) -> str:
        return self.target_ref


@dataclass(frozen=True)
class ActionResultProposal:
    """Terminal status reported by an action owner."""

    status: str = "completed"
    summary: str = ""
    next_action_ids: tuple[str, ...] = ()
    partition_cursor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", str(self.status).lower())
        object.__setattr__(self, "summary", str(self.summary))
        object.__setattr__(
            self, "next_action_ids", tuple(str(item) for item in self.next_action_ids)
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ActionResultProposal":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            status=str(value.get("status", "completed")),
            summary=str(value.get("summary", "")),
            next_action_ids=tuple(
                str(item) for item in _tuple(value.get("next_action_ids"))
            ),
            partition_cursor=value.get("partition_cursor"),
        )


@dataclass(frozen=True)
class EvidenceProposal:
    """Structured evidence submitted by a subagent."""

    action_id: str
    base_version: int
    nodes: tuple[NodeProposal, ...] = ()
    edges: tuple[EdgeProposal, ...] = ()
    action_result: ActionResultProposal = field(default_factory=ActionResultProposal)
    proposal_id: str = field(default_factory=lambda: f"proposal:{uuid4().hex}")
    created_by_sub_traj: int = 0
    created_by_role: Literal["main", "subagent", "system"] = "subagent"
    main_turn: int = 0
    tool_result_refs: tuple[str, ...] = ()

    @classmethod
    def from_dict(
        cls, value: Any, *, action_id: str | None = None, base_version: int = 0
    ) -> "EvidenceProposal":
        if isinstance(value, cls):
            return value
        value = value or {}
        nodes = tuple(
            NodeProposal(
                client_ref=str(item.get("client_ref", item.get("ref", f"node_{idx}"))),
                kind=item.get("kind", "candidate"),
                canonical_key=str(item.get("canonical_key", item.get("key", ""))),
                payload=dict(item.get("payload", {})),
                status=item.get("status", EvidenceStatus.PROPOSED.value),
                confidence=item.get("confidence"),
                tags=tuple(str(tag) for tag in _tuple(item.get("tags"))),
                tool_result_refs=tuple(
                    str(ref) for ref in _tuple(item.get("tool_result_refs"))
                ),
            )
            for idx, item in enumerate(value.get("nodes", []))
        )
        edges = tuple(
            EdgeProposal(
                source_ref=str(item.get("source_ref", item.get("from", ""))),
                relation=str(item.get("relation", item.get("type", ""))),
                target_ref=str(item.get("target_ref", item.get("to", ""))),
            )
            for item in value.get("edges", [])
        )
        return cls(
            proposal_id=str(value.get("proposal_id", f"proposal:{uuid4().hex}")),
            action_id=str(value.get("action_id", action_id or "")),
            base_version=int(value.get("base_version", base_version)),
            nodes=nodes,
            edges=edges,
            action_result=ActionResultProposal.from_dict(value.get("action_result")),
            created_by_sub_traj=int(value.get("created_by_sub_traj", 0)),
            created_by_role=str(value.get("created_by_role", "subagent")),  # type: ignore[arg-type]
            main_turn=int(value.get("main_turn", 0)),
            tool_result_refs=tuple(
                str(ref) for ref in _tuple(value.get("tool_result_refs"))
            ),
        )


@dataclass(frozen=True)
class ActionBudget:
    """Resource limits associated with an action."""

    max_turns: int = 20
    max_tool_calls: int = 20
    max_tokens: int = 8192

    @classmethod
    def from_dict(cls, value: Any) -> "ActionBudget":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            max_turns=max(1, int(value.get("max_turns", 20))),
            max_tool_calls=max(1, int(value.get("max_tool_calls", 20))),
            max_tokens=max(1, int(value.get("max_tokens", 8192))),
        )


@dataclass(frozen=True)
class ActionNode:
    """A bounded unit of work in the Activation DAG."""

    action_id: str
    objective: str
    output_contract: dict[str, JsonValue] = field(default_factory=dict)
    assignee_policy: dict[str, JsonValue] = field(default_factory=dict)
    priority: int = 0
    budget: ActionBudget = field(default_factory=ActionBudget)
    state: ActionState = ActionState.DORMANT
    owner_sub_traj: int | None = None
    predecessor_ids: tuple[str, ...] = ()
    guard_ids: tuple[str, ...] = ()
    payload_ids: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, ActionState):
            object.__setattr__(self, "state", ActionState(str(self.state).lower()))
        object.__setattr__(self, "predecessor_ids", tuple(self.predecessor_ids))
        object.__setattr__(self, "guard_ids", tuple(self.guard_ids))
        object.__setattr__(self, "payload_ids", tuple(self.payload_ids))

    @classmethod
    def from_dict(
        cls, value: Any, *, default_state: ActionState = ActionState.DORMANT
    ) -> "ActionNode":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            action_id=str(value.get("action_id", value.get("id", ""))),
            objective=str(value.get("objective", value.get("prompt", ""))),
            output_contract=dict(value.get("output_contract", {})),
            assignee_policy=dict(value.get("assignee_policy", {})),
            priority=int(value.get("priority", 0)),
            budget=ActionBudget.from_dict(value.get("budget")),
            state=ActionState(str(value.get("state", default_state.value))),
            owner_sub_traj=value.get("owner_sub_traj"),
            predecessor_ids=tuple(
                str(item)
                for item in _tuple(
                    value.get("predecessor_ids", value.get("predecessors"))
                )
            ),
            guard_ids=tuple(
                str(item)
                for item in _tuple(value.get("guard_ids", value.get("guards")))
            ),
            payload_ids=tuple(
                str(item)
                for item in _tuple(value.get("payload_ids", value.get("payloads")))
            ),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class GateNode:
    """Deterministic condition node."""

    gate_id: str
    condition: dict[str, JsonValue]
    satisfied: bool = False
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any) -> "GateNode":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            gate_id=str(value.get("gate_id", value.get("id", ""))),
            condition=dict(value.get("condition", value.get("when", {}))),
            satisfied=bool(value.get("satisfied", False)),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class PayloadNode:
    """Selector recipe used to materialize an activation packet."""

    payload_id: str
    selector: dict[str, JsonValue] = field(default_factory=dict)
    projection: dict[str, JsonValue] = field(default_factory=dict)
    delivery_policy: DeliveryPolicy | str = DeliveryPolicy.ON_READY
    max_tokens: int = 512
    required: bool = True
    # Phase 2 materialization fields.  They are optional so the retained v1
    # plan/compiler schemas remain deserializable.
    target_node_id: str | None = None
    seed_ref: str | None = None
    focus_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    graph_version: int = 0
    token_count: int = 0
    retrieval_metadata: dict[str, JsonValue] = field(default_factory=dict)
    body: dict[str, JsonValue] = field(default_factory=dict)
    page_index: int = 0
    page_count: int = 1

    @classmethod
    def from_dict(cls, value: Any) -> "PayloadNode":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            payload_id=str(value.get("payload_id", value.get("id", ""))),
            selector=dict(value.get("selector", {})),
            projection=dict(value.get("projection", {})),
            delivery_policy=str(
                value.get("delivery_policy", DeliveryPolicy.ON_READY.value)
            ),
            max_tokens=max(1, int(value.get("max_tokens", 512))),
            required=bool(value.get("required", True)),
            target_node_id=value.get("target_node_id"),
            seed_ref=value.get("seed_ref"),
            focus_refs=tuple(str(item) for item in _tuple(value.get("focus_refs"))),
            evidence_refs=tuple(
                str(item) for item in _tuple(value.get("evidence_refs"))
            ),
            graph_version=int(value.get("graph_version", 0)),
            token_count=max(0, int(value.get("token_count", 0))),
            retrieval_metadata=dict(value.get("retrieval_metadata", {})),
            body=dict(value.get("body", {})),
            page_index=max(0, int(value.get("page_index", 0))),
            page_count=max(1, int(value.get("page_count", 1))),
        )


@dataclass(frozen=True)
class JoinNode:
    """Join node for future multi-branch templates."""

    join_id: str
    policy: str = "all_of"
    input_ids: tuple[str, ...] = ()
    state: str = "waiting"

    @classmethod
    def from_dict(cls, value: Any) -> "JoinNode":
        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            join_id=str(value.get("join_id", value.get("id", ""))),
            policy=str(value.get("policy", "all_of")),
            input_ids=tuple(
                str(item)
                for item in _tuple(value.get("input_ids", value.get("inputs")))
            ),
            state=str(value.get("state", "waiting")),
        )


@dataclass(frozen=True)
class AuditNode:
    """Completion audit node."""

    audit_id: str
    policy: dict[str, JsonValue] = field(default_factory=dict)
    state: str = "dormant"
    attempt: int = 0
    graph_version: int = 0
    covered_action_seq: int = 0
    payload_ids: tuple[str, ...] = ()
    outcome: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any) -> "AuditNode":
        """Build an audit node from a JSON-compatible planner payload."""

        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            audit_id=str(value.get("audit_id", value.get("id", ""))),
            policy=dict(value.get("policy", {})),
            state=str(value.get("state", "dormant")),
            attempt=max(0, int(value.get("attempt", 0))),
            graph_version=int(value.get("graph_version", 0)),
            covered_action_seq=int(value.get("covered_action_seq", 0)),
            payload_ids=tuple(str(item) for item in _tuple(value.get("payload_ids"))),
            outcome=dict(value.get("outcome", {})),
        )


@dataclass(frozen=True)
class RenderNode:
    """Deterministic output renderer node."""

    render_id: str
    answer_kind: AnswerKind = "item"
    state: str = "dormant"
    attempt: int = 0
    graph_version: int = 0
    payload_ids: tuple[str, ...] = ()
    page_index: int = 0
    page_count: int = 1
    outcome: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any, *, answer_kind: AnswerKind = "item") -> "RenderNode":
        """Build a renderer node from a JSON-compatible planner payload."""

        if isinstance(value, cls):
            return value
        value = value or {}
        return cls(
            render_id=str(value.get("render_id", value.get("id", ""))),
            answer_kind=str(value.get("answer_kind", answer_kind)),  # type: ignore[arg-type]
            state=str(value.get("state", "dormant")),
            attempt=max(0, int(value.get("attempt", 0))),
            graph_version=int(value.get("graph_version", 0)),
            payload_ids=tuple(str(item) for item in _tuple(value.get("payload_ids"))),
            page_index=max(0, int(value.get("page_index", 0))),
            page_count=max(1, int(value.get("page_count", 1))),
            outcome=dict(value.get("outcome", {})),
        )


@dataclass(frozen=True)
class ActivationEdge:
    """Control/data edge in the Activation DAG."""

    edge_id: str
    source_id: str
    target_id: str
    relation: ActivationEdgeType | str
    outcome: str | None = None
    required: bool = True
    created_at_version: int = 0


@dataclass(frozen=True)
class TaskPlanProposal:
    """Main-agent proposal compiled into canonical task/activation state."""

    contract: TaskContract | dict[str, JsonValue]
    actions: tuple[ActionNode, ...] = ()
    gates: tuple[GateNode, ...] = ()
    payloads: tuple[PayloadNode, ...] = ()
    joins: tuple[JoinNode, ...] = ()
    audits: tuple[AuditNode, ...] = ()
    renders: tuple[RenderNode, ...] = ()
    edges: tuple[ActivationEdge, ...] = ()
    anchor_entities: tuple[NodeProposal, ...] = ()
    proposal_id: str = field(default_factory=lambda: f"plan:{uuid4().hex}")

    @classmethod
    def from_dict(
        cls, value: Any, *, question: str, answer_type: str = "item"
    ) -> "TaskPlanProposal":
        if isinstance(value, cls):
            return value
        value = value or {}
        contract_value = value.get("contract", value)
        contract = TaskContract.from_dict(
            contract_value,
            question=question,
        )
        return cls(
            proposal_id=str(value.get("proposal_id", f"plan:{uuid4().hex}")),
            contract=contract,
            actions=tuple(
                ActionNode.from_dict(item) for item in value.get("actions", [])
            ),
            gates=tuple(GateNode.from_dict(item) for item in value.get("gates", [])),
            payloads=tuple(
                PayloadNode.from_dict(item) for item in value.get("payloads", [])
            ),
            joins=tuple(JoinNode.from_dict(item) for item in value.get("joins", [])),
            audits=tuple(AuditNode.from_dict(item) for item in value.get("audits", [])),
            renders=tuple(
                RenderNode.from_dict(item, answer_kind=answer_type)  # type: ignore[arg-type]
                for item in value.get("renders", [])
            ),
            edges=tuple(
                item
                if isinstance(item, ActivationEdge)
                else ActivationEdge(
                    edge_id=str(item.get("edge_id", item.get("id", f"edge:{idx}"))),
                    source_id=str(item.get("source_id", item.get("source", ""))),
                    target_id=str(item.get("target_id", item.get("target", ""))),
                    relation=str(item.get("relation", "precedes")),
                    outcome=item.get("outcome"),
                    required=bool(item.get("required", True)),
                )
                for idx, item in enumerate(value.get("edges", []))
            ),
            anchor_entities=tuple(
                item
                if isinstance(item, NodeProposal)
                else NodeProposal(
                    client_ref=str(
                        item.get("client_ref", item.get("ref", f"entity_{idx}"))
                    ),
                    kind=EvidenceKind.ENTITY,
                    canonical_key=str(item.get("canonical_key", item.get("key", ""))),
                    payload=dict(item.get("payload", {})),
                )
                for idx, item in enumerate(
                    value.get("anchor_entities", value.get("anchors", []))
                )
            ),
        )


@dataclass(frozen=True)
class ActivationEvent:
    """Turn-boundary notification for an activated action."""

    event_type: str
    event_id: str
    action_id: str
    graph_version: int
    reason: dict[str, JsonValue] = field(default_factory=dict)
    evidence_refs: tuple[dict[str, JsonValue], ...] = ()
    allowed_reads: tuple[str, ...] = ()
    expires_after_version: int | None = None
    payload: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphDelta:
    """Changed canonical references returned by an evidence commit."""

    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()
    conflict_ids: tuple[str, ...] = ()


class GraphEventType(str, Enum):
    """Append-only event types emitted by the Phase 1 graph."""

    ADD_NODE = "add_node"
    ADD_EDGE = "add_edge"
    UPDATE_METADATA = "update_metadata"
    MERGE_ENTITY = "merge_entity"
    PROMOTE_CLAIM = "promote_claim"
    RESOLVE_CONFLICT = "resolve_conflict"
    RETIRE_NODE = "retire_node"
    RETIRE_EDGE = "retire_edge"
    POTENTIAL_CONFLICT = "potential_conflict"
    BOOTSTRAP_ENTITY = "bootstrap_entity"
    TOOL_RESULT = "tool_result"
    CREATE_ACTION = "create_action"
    MATERIALIZE_PAYLOAD = "materialize_payload"
    CREATE_AUDIT = "create_audit"
    AUDIT_OUTCOME = "audit_outcome"
    CREATE_RENDER = "create_render"
    FORMAT_RETRY = "format_retry"


@dataclass(frozen=True)
class GraphEvent:
    """Immutable append-only record for one canonical graph mutation."""

    event_id: str
    event_type: GraphEventType | str
    graph_version: int
    main_turn: int = 0
    actor_role: Literal["main", "subagent", "system"] = "system"
    action_id: str | None = None
    sub_traj_id: int = 0
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    tool_result_refs: tuple[str, ...] = ()
    payload: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, GraphEventType):
            object.__setattr__(
                self, "event_type", GraphEventType(str(self.event_type).lower())
            )
        if self.actor_role not in {"main", "subagent", "system"}:
            raise ValueError(f"Unsupported event actor role: {self.actor_role!r}")
        object.__setattr__(self, "node_ids", tuple(str(item) for item in self.node_ids))
        object.__setattr__(self, "edge_ids", tuple(str(item) for item in self.edge_ids))
        object.__setattr__(
            self,
            "tool_result_refs",
            tuple(str(item) for item in self.tool_result_refs),
        )
        object.__setattr__(self, "payload", dict(self.payload or {}))


@dataclass(frozen=True)
class ToolResultRecord:
    """Bounded provenance metadata for one external search/access result."""

    tool_result_id: str
    tool_name: str
    action_id: str
    sub_traj_id: int
    main_turn: int
    result_hash: str
    query: str | None = None
    url: str | None = None
    result: str = ""
    success: bool = True


@dataclass(frozen=True)
class CommitResult:
    """Result of an atomic graph commit."""

    proposal_id: str
    graph_version: int
    delta: GraphDelta
    transitions: tuple[dict[str, JsonValue], ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class GapReport:
    """Structured completion gap report."""

    missing_fields: tuple[str, ...] = ()
    open_conflicts: tuple[str, ...] = ()
    unclosed_partitions: tuple[str, ...] = ()
    suggested_action_templates: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditResult:
    """Mechanical completion audit result."""

    passed: bool
    status: str
    fact_refs: tuple[str, ...] = ()
    gap_report: GapReport = field(default_factory=GapReport)


@dataclass(frozen=True)
class GraphConfig:
    """Runtime limits and verification policy for graph-memory v2."""

    enabled: bool = False
    schema_version: str = "v2"
    condition_dsl_version: str = "v2"
    max_nodes: int = 20_000
    max_edges: int = 60_000
    max_actions: int = 5_000
    max_dynamic_expansions: int = 4_000
    max_graph_versions: int = 20_000
    max_notification_tokens: int = 512
    max_selected_evidence_tokens: int = 2_048
    max_delta_events_per_turn: int = 8
    planner_recent_delta_versions: int = 3
    event_ttl_versions: int = 20
    max_concurrent_actions: int = 32
    action_retry_limit: int = 2
    require_source_for_verified_claim: bool = True
    min_independent_sources: int = 1
    allow_worker_fact_proposal: bool = False
    deterministic_render: bool = True
    eval_allow_direct_answer_fallback: bool = False
    eval_direct_answer_fallback_max_tokens: int = 1_024
    reject_mixed_tool_phases: bool = True
    log_snapshot: bool = True
    max_snapshot_nodes: int = 2_000
    snapshot_include_source_excerpt: bool = False
    max_events: int = 60_000
    max_read_tokens: int = 2_048
    entity_bootstrap_enabled: bool = True
    entity_bootstrap_max_new_tokens: int = 512
    require_tool_provenance: bool = True
    allow_worker_read_mem: bool = False
    require_claim_next_turn_verification: bool = True
    max_pending_claims: int = 2_000
    max_pending_conflicts: int = 2_000
    # Phase 2 deterministic retrieval and payload budgets.
    embedding_backend: str = "hash"
    embedding_dim: int = 128
    payload_top_k: int = 4
    payload_max_distance: int = 1
    max_payload_nodes: int = 64
    max_payload_tokens: int = 2_048
    max_source_excerpt_tokens: int = 256
    payload_include_source_excerpt: bool = True
    # Phase 4 independent audit/render budgets and safety switches.
    audit_enabled: bool = True
    render_enabled: bool = True
    format_retry_enabled: bool = True
    max_audit_attempts: int = 3
    max_render_attempts: int = 3
    audit_best_effort: bool = True
    require_audit_pass: bool = True
    max_render_page_rows: int = 32

    @classmethod
    def from_config(cls, value: Any) -> "GraphConfig":
        if isinstance(value, cls):
            return value
        value = value or {}
        fields = {
            key: value.get(key, getattr(cls, key)) for key in cls.__dataclass_fields__
        }
        for key in (
            "max_nodes",
            "max_edges",
            "max_actions",
            "max_dynamic_expansions",
            "max_graph_versions",
            "max_notification_tokens",
            "max_selected_evidence_tokens",
            "max_delta_events_per_turn",
            "planner_recent_delta_versions",
            "event_ttl_versions",
            "max_concurrent_actions",
            "action_retry_limit",
            "min_independent_sources",
            "max_snapshot_nodes",
            "eval_direct_answer_fallback_max_tokens",
            "max_events",
            "max_read_tokens",
            "entity_bootstrap_max_new_tokens",
            "max_pending_claims",
            "max_pending_conflicts",
            "embedding_dim",
            "payload_top_k",
            "payload_max_distance",
            "max_payload_nodes",
            "max_payload_tokens",
            "max_source_excerpt_tokens",
            "max_audit_attempts",
            "max_render_attempts",
            "max_render_page_rows",
        ):
            fields[key] = int(fields[key])
        for key in (
            "enabled",
            "require_source_for_verified_claim",
            "allow_worker_fact_proposal",
            "deterministic_render",
            "eval_allow_direct_answer_fallback",
            "reject_mixed_tool_phases",
            "log_snapshot",
            "snapshot_include_source_excerpt",
            "entity_bootstrap_enabled",
            "require_tool_provenance",
            "allow_worker_read_mem",
            "require_claim_next_turn_verification",
            "payload_include_source_excerpt",
            "audit_enabled",
            "render_enabled",
            "format_retry_enabled",
            "audit_best_effort",
            "require_audit_pass",
        ):
            fields[key] = bool(fields[key])
        fields["embedding_backend"] = str(fields["embedding_backend"])
        return cls(**fields)
