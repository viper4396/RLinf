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

"""In-memory, trajectory-local state for WideSeek-R2 graph memory."""

from __future__ import annotations

import asyncio
import copy
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionNode,
    ActionState,
    ActivationEdge,
    ActivationEdgeType,
    ActivationEvent,
    ActivationNodeKind,
    AuditNode,
    EvidenceEdge,
    EvidenceKind,
    EvidenceNode,
    EvidenceStatus,
    GateNode,
    GraphConfig,
    JoinNode,
    PayloadNode,
    RenderNode,
    TaskContract,
    TaskPlanProposal,
)


class GraphStateError(RuntimeError):
    """Raised when a graph state transition is invalid."""


@dataclass
class EvidenceGraph:
    """Mutable canonical Evidence Graph owned by one rollout."""

    version: int = 0
    nodes: dict[str, EvidenceNode] = field(default_factory=dict)
    edges: dict[str, EvidenceEdge] = field(default_factory=dict)
    canonical_index: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def canonical_lookup_key(kind: EvidenceKind | str, canonical_key: str) -> str:
        return f"{EvidenceKind.coerce(kind).value}:{canonical_key}"

    def get_by_canonical(
        self, canonical_key: str, kind: EvidenceKind | str | None = None
    ) -> EvidenceNode | None:
        if kind is not None:
            node_id = self.canonical_index.get(
                self.canonical_lookup_key(kind, canonical_key)
            )
            return self.nodes.get(node_id) if node_id else None
        matches = {
            node_id
            for key, node_id in self.canonical_index.items()
            if key.endswith(f":{canonical_key}")
        }
        # A plain alias is only safe when the canonical key is unambiguous
        # across evidence kinds.  Typed lookups above remain authoritative.
        if len(matches) == 1:
            return self.nodes[next(iter(matches))]
        if not matches:
            node_id = self.canonical_index.get(canonical_key)
            return self.nodes.get(node_id) if node_id else None
        return None

    def add_node(self, node: EvidenceNode) -> tuple[EvidenceNode, bool]:
        """Insert a node, merging by canonical key.

        Returns the canonical node and whether it was newly inserted.
        """

        lookup_key = self.canonical_lookup_key(node.kind, node.canonical_key)
        existing_id = self.canonical_index.get(lookup_key)
        if existing_id is not None:
            return self.nodes[existing_id], False
        self.nodes[node.node_id] = node
        self.canonical_index[lookup_key] = node.node_id
        # A plain-key alias is useful for task-local references when there is
        # no ambiguity, while the typed key remains canonical.
        self.canonical_index.setdefault(node.canonical_key, node.node_id)
        return node, True

    def replace_node(self, node: EvidenceNode) -> None:
        if node.node_id not in self.nodes:
            raise GraphStateError(f"Unknown evidence node {node.node_id!r}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: EvidenceEdge) -> tuple[EvidenceEdge, bool]:
        if edge.edge_id in self.edges:
            return self.edges[edge.edge_id], False
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise GraphStateError(
                f"Evidence edge endpoints do not exist: {edge.source_id!r} -> {edge.target_id!r}"
            )
        # Same relationship is idempotent even if a client uses a new edge ID.
        for existing in self.edges.values():
            if (
                existing.source_id == edge.source_id
                and existing.relation == edge.relation
                and existing.target_id == edge.target_id
            ):
                return existing, False
        self.edges[edge.edge_id] = edge
        return edge, True

    def iter_kind(self, kind: EvidenceKind | str) -> list[EvidenceNode]:
        kind = EvidenceKind.coerce(kind)
        return [node for node in self.nodes.values() if node.kind == kind]

    def incoming(
        self, target_id: str, relation: str | None = None
    ) -> list[EvidenceEdge]:
        return [
            edge
            for edge in self.edges.values()
            if edge.target_id == target_id
            and (relation is None or edge.relation == relation)
        ]

    def outgoing(
        self, source_id: str, relation: str | None = None
    ) -> list[EvidenceEdge]:
        return [
            edge
            for edge in self.edges.values()
            if edge.source_id == source_id
            and (relation is None or edge.relation == relation)
        ]

    def facts_for_claim(self, claim_id: str) -> list[EvidenceNode]:
        fact_ids = {edge.target_id for edge in self.outgoing(claim_id, "VERIFIED_AS")}
        return [
            self.nodes[fact_id]
            for fact_id in fact_ids
            if fact_id in self.nodes and self.nodes[fact_id].kind == EvidenceKind.FACT
        ]


@dataclass
class ActivationDAG:
    """Typed Activation DAG and its reverse dependency indexes."""

    nodes: dict[str, tuple[ActivationNodeKind, Any]] = field(default_factory=dict)
    actions: dict[str, ActionNode] = field(default_factory=dict)
    gates: dict[str, GateNode] = field(default_factory=dict)
    payloads: dict[str, PayloadNode] = field(default_factory=dict)
    joins: dict[str, JoinNode] = field(default_factory=dict)
    audits: dict[str, AuditNode] = field(default_factory=dict)
    renders: dict[str, RenderNode] = field(default_factory=dict)
    edges: dict[str, ActivationEdge] = field(default_factory=dict)
    incoming: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    outgoing: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, node: Any) -> None:
        if isinstance(node, ActionNode):
            kind = ActivationNodeKind.ACTION
            self.actions[node.action_id] = node
            node_id = node.action_id
        elif isinstance(node, GateNode):
            kind = ActivationNodeKind.GATE
            self.gates[node.gate_id] = node
            node_id = node.gate_id
        elif isinstance(node, PayloadNode):
            kind = ActivationNodeKind.PAYLOAD
            self.payloads[node.payload_id] = node
            node_id = node.payload_id
        elif isinstance(node, JoinNode):
            kind = ActivationNodeKind.JOIN
            self.joins[node.join_id] = node
            node_id = node.join_id
        elif isinstance(node, AuditNode):
            kind = ActivationNodeKind.AUDIT
            self.audits[node.audit_id] = node
            node_id = node.audit_id
        elif isinstance(node, RenderNode):
            kind = ActivationNodeKind.RENDER
            self.renders[node.render_id] = node
            node_id = node.render_id
        else:
            raise TypeError(f"Unsupported activation node: {type(node)!r}")
        if node_id in self.nodes:
            raise GraphStateError(f"Duplicate activation node {node_id!r}")
        self.nodes[node_id] = (kind, node)

    def replace_node(self, node: Any) -> None:
        node_id = self.node_id(node)
        if node_id not in self.nodes:
            raise GraphStateError(f"Unknown activation node {node_id!r}")
        kind, _ = self.nodes[node_id]
        self.nodes[node_id] = (kind, node)
        if kind == ActivationNodeKind.ACTION:
            self.actions[node_id] = node
        elif kind == ActivationNodeKind.GATE:
            self.gates[node_id] = node
        elif kind == ActivationNodeKind.PAYLOAD:
            self.payloads[node_id] = node
        elif kind == ActivationNodeKind.JOIN:
            self.joins[node_id] = node
        elif kind == ActivationNodeKind.AUDIT:
            self.audits[node_id] = node
        elif kind == ActivationNodeKind.RENDER:
            self.renders[node_id] = node

    @staticmethod
    def node_id(node: Any) -> str:
        for attribute in (
            "action_id",
            "gate_id",
            "payload_id",
            "join_id",
            "audit_id",
            "render_id",
        ):
            if hasattr(node, attribute):
                return str(getattr(node, attribute))
        raise TypeError(f"Cannot determine activation node ID for {node!r}")

    def add_edge(self, edge: ActivationEdge) -> None:
        relation = (
            edge.relation
            if isinstance(edge.relation, ActivationEdgeType)
            else ActivationEdgeType(str(edge.relation))
        )
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise GraphStateError(
                f"Activation edge endpoints do not exist: {edge.source_id!r} -> {edge.target_id!r}"
            )
        if edge.edge_id in self.edges:
            return
        normalized = ActivationEdge(
            edge_id=edge.edge_id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=relation,
            outcome=edge.outcome,
            required=edge.required,
            created_at_version=edge.created_at_version,
        )
        self.edges[edge.edge_id] = normalized
        self.incoming[edge.target_id].add(edge.edge_id)
        self.outgoing[edge.source_id].add(edge.edge_id)

    def edges_to(
        self, node_id: str, relation: ActivationEdgeType | str | None = None
    ) -> list[ActivationEdge]:
        return [
            self.edges[edge_id]
            for edge_id in self.incoming.get(node_id, set())
            if relation is None or self.edges[edge_id].relation == relation
        ]

    def edges_from(
        self, node_id: str, relation: ActivationEdgeType | str | None = None
    ) -> list[ActivationEdge]:
        return [
            self.edges[edge_id]
            for edge_id in self.outgoing.get(node_id, set())
            if relation is None or self.edges[edge_id].relation == relation
        ]

    def validate_acyclic(self) -> None:
        """Reject cycles in all control edges.

        ``DELIVERS`` is data flow and does not make an action wait for itself;
        it is therefore ignored for cycle detection.  All other relations are
        scheduler control edges.
        """

        adjacency: dict[str, set[str]] = defaultdict(set)
        indegree = dict.fromkeys(self.nodes, 0)
        for edge in self.edges.values():
            if edge.relation == ActivationEdgeType.DELIVERS:
                continue
            if edge.target_id not in adjacency[edge.source_id]:
                adjacency[edge.source_id].add(edge.target_id)
                indegree[edge.target_id] += 1
        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            node_id = queue.popleft()
            visited += 1
            for target_id in adjacency.get(node_id, set()):
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    queue.append(target_id)
        if visited != len(indegree):
            cycle_nodes = sorted(
                node_id for node_id, degree in indegree.items() if degree > 0
            )
            raise GraphStateError(f"Activation DAG contains a cycle: {cycle_nodes}")

    def predecessor_ids(self, action_id: str) -> tuple[str, ...]:
        action = self.actions.get(action_id)
        edge_ids = {
            edge.source_id
            for edge in self.edges_to(action_id)
            if edge.relation
            in {
                ActivationEdgeType.PRECEDES,
                ActivationEdgeType.ON_OUTCOME,
                ActivationEdgeType.JOINS_INTO,
            }
        }
        if action is not None:
            edge_ids.update(action.predecessor_ids)
        return tuple(sorted(edge_ids))

    def guard_ids(self, action_id: str) -> tuple[str, ...]:
        guards = {
            edge.source_id
            for edge in self.edges_to(action_id, ActivationEdgeType.GUARDS)
        }
        action = self.actions.get(action_id)
        if action is not None:
            guards.update(action.guard_ids)
        return tuple(sorted(guards))

    def payload_ids(self, action_id: str) -> tuple[str, ...]:
        payloads = {
            edge.source_id
            for edge in self.edges_to(action_id, ActivationEdgeType.DELIVERS)
        }
        action = self.actions.get(action_id)
        if action is not None:
            payloads.update(action.payload_ids)
        return tuple(sorted(payloads))


_graph_runtime_ctx: ContextVar["GraphRuntime | None"] = ContextVar(
    "wideseek_r2_graph_runtime", default=None
)
_current_action_ctx: ContextVar[str | None] = ContextVar(
    "wideseek_r2_current_action", default=None
)
_current_sub_traj_ctx: ContextVar[int] = ContextVar(
    "wideseek_r2_current_sub_traj", default=0
)
_current_phase_ctx: ContextVar[str | None] = ContextVar(
    "wideseek_r2_graph_tool_phase", default=None
)


def get_graph_runtime() -> "GraphRuntime":
    """Return the current trajectory-local runtime or raise a clear error."""

    runtime = _graph_runtime_ctx.get()
    if runtime is None:
        raise GraphStateError("No graph runtime is active in this context")
    return runtime


def get_current_action() -> str | None:
    """Return the action owning the current worker context."""

    return _current_action_ctx.get()


def get_current_sub_traj() -> int:
    """Return the current worker sub-trajectory for lineage attribution."""

    return _current_sub_traj_ctx.get()


def get_current_phase() -> str | None:
    """Return the current turn's graph/external tool phase."""

    return _current_phase_ctx.get()


@dataclass
class GraphRuntime:
    """All canonical graph state for exactly one rollout trajectory."""

    question: str
    answer_type: str
    config: GraphConfig = field(default_factory=GraphConfig)
    language: str = "en"
    format_requirements: dict[str, Any] = field(default_factory=dict)
    budget: int | None = None
    evidence_graph: EvidenceGraph = field(default_factory=EvidenceGraph)
    activation_dag: ActivationDAG = field(default_factory=ActivationDAG)
    contract: TaskContract | None = None
    version: int = 0
    plan_version: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    event_queues: dict[str, deque[ActivationEvent]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    emitted_event_keys: set[str] = field(default_factory=set)
    events_by_key: dict[str, ActivationEvent] = field(default_factory=dict)
    delivered_event_keys: set[str] = field(default_factory=set)
    planner_seen_event_ids: set[str] = field(default_factory=set)
    action_acl: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    action_acl_expiry: dict[str, int] = field(default_factory=dict)
    action_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    graph_local_results: dict[int, str] = field(default_factory=dict)
    covered_partitions: set[str] = field(default_factory=set)
    remaining_budget: int = 0
    finish_requested: bool = False
    finish_reason: str = ""
    last_audit: Any | None = None
    phase: str | None = None
    phase_history: list[str] = field(default_factory=list)
    last_error: str | None = None

    # These aliases let the condition evaluator work with plain enum values
    # without importing ActionState in every call site.
    action_state_completed = ActionState.COMPLETED
    action_state_consumed = ActionState.CONSUMED

    def __post_init__(self) -> None:
        self.answer_type = str(self.answer_type or "table").lower()
        self.language = str(self.language or "en")
        self.format_requirements = copy.deepcopy(self.format_requirements or {})
        if self.budget is not None:
            self.budget = max(0, min(int(self.budget), self.config.max_actions))
            self.remaining_budget = min(max(0, self.remaining_budget), self.budget)
        if self.remaining_budget <= 0 and self.budget is None:
            self.remaining_budget = self.config.max_actions
        elif self.remaining_budget <= 0:
            self.remaining_budget = self.budget

    @classmethod
    def bootstrap(
        cls,
        *,
        question: str,
        answer_type: str | None,
        config: GraphConfig | dict[str, Any] | None = None,
        language: str = "en",
        format_requirements: dict[str, Any] | None = None,
        budget: int | None = None,
    ) -> "GraphRuntime":
        """Create an empty, trajectory-local graph and fixed DAG skeleton."""

        normalized_type = str(answer_type or "table").lower()
        if normalized_type not in {"item", "set", "list", "table"}:
            raise ValueError(
                f"Unsupported answer_type {answer_type!r}; expected item, set, list, or table"
            )
        runtime = cls(
            question=question,
            answer_type=normalized_type,
            config=GraphConfig.from_config(config),
            language=language,
            format_requirements=format_requirements or {},
            budget=budget,
        )
        runtime._bootstrap_activation_skeleton()
        return runtime

    def _bootstrap_activation_skeleton(self) -> None:
        dag = self.activation_dag
        dag.add_node(
            PayloadNode(
                payload_id="payload:task_context",
                selector={"task_context": True},
                projection={"question": True, "answer_type": True},
                max_tokens=self.config.max_notification_tokens,
                required=True,
            )
        )
        dag.add_node(
            ActionNode(
                action_id="action:plan_task",
                objective="Compile a Task Contract and an item dependency plan.",
                state=ActionState.READY,
                payload_ids=("payload:task_context",),
            )
        )
        dag.add_node(
            GateNode(
                gate_id="gate:contract_valid",
                condition={"op": "false"},
            )
        )
        dag.add_node(
            ActionNode(
                action_id="action:initial_frontier",
                objective="Dispatch the first ready discovery action.",
                state=ActionState.DORMANT,
            )
        )
        dag.add_node(
            GateNode(
                gate_id="gate:finish_eligible",
                condition={"op": "false"},
            )
        )
        dag.add_node(AuditNode(audit_id="audit:completion"))
        dag.add_node(
            RenderNode(
                render_id="render:final",
                answer_kind=self.answer_type,  # type: ignore[arg-type]
            )
        )
        dag.add_edge(
            ActivationEdge(
                edge_id="edge:task_context_to_plan",
                source_id="payload:task_context",
                target_id="action:plan_task",
                relation=ActivationEdgeType.DELIVERS,
            )
        )
        dag.add_edge(
            ActivationEdge(
                edge_id="edge:contract_to_frontier",
                source_id="gate:contract_valid",
                target_id="action:initial_frontier",
                relation=ActivationEdgeType.GUARDS,
            )
        )
        dag.add_edge(
            ActivationEdge(
                edge_id="edge:finish_to_audit",
                source_id="gate:finish_eligible",
                target_id="audit:completion",
                relation=ActivationEdgeType.GUARDS,
            )
        )
        dag.add_edge(
            ActivationEdge(
                edge_id="edge:audit_to_render",
                source_id="audit:completion",
                target_id="render:final",
                relation=ActivationEdgeType.ON_OUTCOME,
                outcome="pass",
            )
        )
        dag.validate_acyclic()

    def set_phase(self, phase: str | None) -> None:
        """Set the current model turn's tool phase.

        ``graph`` and ``external`` are mutually exclusive when configured;
        resetting to ``None`` happens at every turn boundary. The guard is
        ContextVar-local so concurrent worker turns do not interfere with one
        another while sharing the canonical runtime.
        """

        if phase is None:
            _current_phase_ctx.set(None)
            self.phase = None
            return
        phase = str(phase)
        current_phase = _current_phase_ctx.get()
        if self.config.reject_mixed_tool_phases and current_phase not in {
            None,
            phase,
        }:
            raise GraphStateError(
                f"Mixed graph tool phases are not allowed: {current_phase!r} and {phase!r}"
            )
        _current_phase_ctx.set(phase)
        self.phase = phase
        self.phase_history.append(phase)

    def begin_turn(self) -> None:
        _current_phase_ctx.set(None)
        self.phase = None

    def task_context(self) -> dict[str, Any]:
        """Return the non-secret bootstrap context exposed to the planner."""

        context = {
            "question": self.question,
            "answer_type": self.answer_type,
            "language": self.language,
            "format_requirements": copy.deepcopy(self.format_requirements),
            "graph_version": self.version,
            "plan_version": self.plan_version,
            "budget": self.config.max_actions if self.budget is None else self.budget,
            "remaining_budget": self.remaining_budget,
        }
        # Deliberately no answer/label/reward/judge fields are accepted here.
        return context

    def install_plan(
        self,
        contract: TaskContract,
        *,
        actions: list[ActionNode] | tuple[ActionNode, ...] = (),
        gates: list[GateNode] | tuple[GateNode, ...] = (),
        payloads: list[PayloadNode] | tuple[PayloadNode, ...] = (),
        joins: list[JoinNode] | tuple[JoinNode, ...] = (),
        audits: list[AuditNode] | tuple[AuditNode, ...] = (),
        renders: list[RenderNode] | tuple[RenderNode, ...] = (),
        edges: list[ActivationEdge] | tuple[ActivationEdge, ...] = (),
    ) -> None:
        """Install a validated plan and activate its initial frontier."""

        self.contract = contract
        self.plan_version += 1
        dag = self.activation_dag
        for node in payloads:
            if node.payload_id not in dag.nodes:
                dag.add_node(node)
        for node in gates:
            if node.gate_id not in dag.nodes:
                dag.add_node(node)
        for node in joins:
            if node.join_id not in dag.nodes:
                dag.add_node(node)
        for node in actions:
            if node.action_id not in dag.nodes:
                dag.add_node(node)
        for node in audits:
            if node.audit_id not in dag.nodes:
                dag.add_node(node)
        for node in renders:
            if node.render_id not in dag.nodes:
                dag.add_node(node)
        for edge in edges:
            if edge.edge_id not in dag.edges:
                dag.add_edge(edge)
        # Plan compilation is the deterministic owner of this internal gate.
        contract_gate = dag.gates.get("gate:contract_valid")
        if contract_gate is not None:
            dag.replace_node(
                GateNode(
                    gate_id=contract_gate.gate_id,
                    condition={"op": "true"},
                    satisfied=True,
                    metadata={"plan_version": self.plan_version},
                )
            )
        plan_action = dag.actions.get("action:plan_task")
        if plan_action is not None:
            dag.replace_node(
                ActionNode(
                    **{
                        **plan_action.__dict__,
                        "state": ActionState.COMPLETED,
                    }
                )
            )
        frontier = dag.actions.get("action:initial_frontier")
        if frontier is not None:
            dag.replace_node(
                ActionNode(**{**frontier.__dict__, "state": ActionState.COMPLETED})
            )
        self.evaluate_activation()

    def evaluate_activation(self) -> list[dict[str, Any]]:
        """Recompute gates and action readiness deterministically."""

        from rlinf.agents.wideseek_r2.graph_memory.conditions import (
            ConditionError,
            evaluate_condition,
        )
        from rlinf.agents.wideseek_r2.graph_memory.selectors import (
            materialize_payload,
        )

        transitions: list[dict[str, Any]] = []
        for gate_id, gate in list(self.activation_dag.gates.items()):
            try:
                satisfied = evaluate_condition(gate.condition, self)
            except ConditionError as exc:
                self.last_error = str(exc)
                satisfied = False
            if satisfied != gate.satisfied:
                self.activation_dag.replace_node(
                    GateNode(
                        gate_id=gate.gate_id,
                        condition=copy.deepcopy(gate.condition),
                        satisfied=satisfied,
                        metadata=dict(gate.metadata),
                    )
                )
                transitions.append(
                    {"node_id": gate_id, "from": gate.satisfied, "to": satisfied}
                )

        for action_id, action in list(self.activation_dag.actions.items()):
            if action.state not in {ActionState.DORMANT, ActionState.BLOCKED}:
                continue
            running_count = sum(
                candidate.state == ActionState.RUNNING
                for candidate in self.activation_dag.actions.values()
            )
            ready_count = sum(
                candidate.state == ActionState.READY
                for candidate in self.activation_dag.actions.values()
            )
            if running_count + ready_count >= self.config.max_concurrent_actions:
                continue
            if self.remaining_budget <= 0:
                continue
            predecessors = self.activation_dag.predecessor_ids(action_id)
            if any(
                not self.activation_dag.actions.get(predecessor_id)
                or self.activation_dag.actions[predecessor_id].state
                not in {ActionState.COMPLETED, ActionState.CONSUMED}
                for predecessor_id in predecessors
            ):
                continue
            guards = self.activation_dag.guard_ids(action_id)
            if any(
                not self.activation_dag.gates.get(gate_id)
                or not self.activation_dag.gates[gate_id].satisfied
                for gate_id in guards
            ):
                continue
            payload_ids = (
                self.activation_dag.payload_ids(action_id) or action.payload_ids
            )
            materialized = []
            blocked = False
            for payload_id in payload_ids:
                payload = self.activation_dag.payloads.get(payload_id)
                if payload is None:
                    blocked = True
                    break
                packet = materialize_payload(self, payload, action_id=action_id)
                if packet is None and payload.required:
                    blocked = True
                    break
                if packet is not None:
                    materialized.append(packet)
            if blocked:
                continue
            ready_action = ActionNode(**{**action.__dict__, "state": ActionState.READY})
            self.activation_dag.replace_node(ready_action)
            transitions.append(
                {
                    "node_id": action_id,
                    "from": action.state.value,
                    "to": ActionState.READY.value,
                }
            )
            if action_id not in {"action:plan_task", "action:initial_frontier"}:
                self._enqueue_activation_event(action_id, materialized)
        return transitions

    def _enqueue_activation_event(
        self, action_id: str, packets: list[dict[str, Any]]
    ) -> ActivationEvent:
        refs_by_id: dict[str, dict[str, Any]] = {}
        allowed_reads: set[str] = set()
        for packet in packets:
            for evidence in packet.get("evidence", []):
                ref = str(evidence.get("ref", ""))
                if ref:
                    refs_by_id.setdefault(ref, evidence)
            allowed_reads.update(packet.get("allowed_reads", []))
        self.action_acl.setdefault(action_id, set()).update(allowed_reads)
        self.action_acl_expiry[action_id] = (
            self.version + self.config.event_ttl_versions
        )
        reason = {
            "action_id": action_id,
            "guard_ids": list(self.activation_dag.guard_ids(action_id)),
            "satisfied_by": sorted(allowed_reads),
        }
        key = f"{action_id}:{self.version}:{','.join(sorted(allowed_reads))}"
        if key in self.emitted_event_keys:
            return self.events_by_key[key]
        event = ActivationEvent(
            event_type="ACTIVATION_READY",
            event_id=f"event:{uuid4().hex}",
            action_id=action_id,
            graph_version=self.version,
            reason=reason,
            evidence_refs=tuple(refs_by_id[ref] for ref in sorted(refs_by_id)),
            allowed_reads=tuple(sorted(allowed_reads)),
            expires_after_version=self.version + self.config.event_ttl_versions,
            payload={"objective": self.activation_dag.actions[action_id].objective},
        )
        self.event_queues[action_id].append(event)
        self.emitted_event_keys.add(key)
        self.events_by_key[key] = event
        return event

    def pending_events(
        self, action_id: str, *, consume: bool = False
    ) -> list[ActivationEvent]:
        """Return action events for injection at a model turn boundary."""

        queue = self.event_queues.get(action_id, deque())
        while (
            queue
            and queue[0].expires_after_version is not None
            and self.version > queue[0].expires_after_version
        ):
            queue.popleft()
        events = list(queue)[: self.config.max_delta_events_per_turn]
        if consume:
            for _ in range(len(events)):
                queue.popleft()
            for event in events:
                self.delivered_event_keys.add(event.event_id)
        return events

    def mark_action_running(
        self, action_id: str, *, owner_sub_traj: int | None = None
    ) -> None:
        action = self.activation_dag.actions.get(action_id)
        if action is None:
            raise GraphStateError(f"Unknown action {action_id!r}")
        if action.state not in {ActionState.READY, ActionState.RUNNING}:
            raise GraphStateError(
                f"Action {action_id!r} is not ready (state={action.state.value})"
            )
        if action.state == ActionState.READY:
            running = sum(
                candidate.state == ActionState.RUNNING
                for candidate in self.activation_dag.actions.values()
            )
            if running >= self.config.max_concurrent_actions:
                raise GraphStateError("Maximum concurrent graph actions exceeded")
            if self.remaining_budget <= 0:
                raise GraphStateError("Graph action budget is exhausted")
        if action.state == ActionState.READY:
            self.remaining_budget = max(0, self.remaining_budget - 1)
        self.activation_dag.replace_node(
            ActionNode(
                **{
                    **action.__dict__,
                    "state": ActionState.RUNNING,
                    "owner_sub_traj": owner_sub_traj,
                }
            )
        )

    def mark_action_completed(
        self,
        action_id: str,
        *,
        status: str = "completed",
        summary: str = "",
        partition: str | None = None,
    ) -> list[dict[str, Any]]:
        action = self.activation_dag.actions.get(action_id)
        if action is None:
            raise GraphStateError(f"Unknown action {action_id!r}")
        target_state = {
            "completed": ActionState.COMPLETED,
            "failed": ActionState.FAILED,
            "blocked": ActionState.BLOCKED,
            "invalidated": ActionState.INVALIDATED,
        }.get(str(status).lower(), ActionState.COMPLETED)
        self.activation_dag.replace_node(
            ActionNode(**{**action.__dict__, "state": target_state})
        )
        self.action_results[action_id] = {"status": status, "summary": summary}
        if partition:
            self.covered_partitions.add(partition)
        return self.evaluate_activation()

    async def submit_task_plan(self, proposal: TaskPlanProposal) -> TaskContract:
        """Compile a main-agent plan through the canonical validator."""

        from rlinf.agents.wideseek_r2.graph_memory.validator import compile_task_plan

        return await compile_task_plan(self, proposal)

    async def commit_evidence(self, proposal: Any):
        """Commit a worker evidence proposal through the canonical validator."""

        from rlinf.agents.wideseek_r2.graph_memory.validator import commit_evidence

        return await commit_evidence(self, proposal)

    def read_evidence(
        self,
        refs: list[str] | tuple[str, ...],
        *,
        action_id: str | None = None,
        fields: list[str] | tuple[str, ...] | None = None,
        since_version: int | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read evidence through the action-scoped selector."""

        from rlinf.agents.wideseek_r2.graph_memory.selectors import (
            read_scoped_evidence,
        )

        if not action_id:
            raise GraphStateError("read_evidence requires an action_id")
        return read_scoped_evidence(
            self,
            action_id=action_id,
            refs=refs,
            fields=fields,
            since_version=since_version,
            max_tokens=max_tokens,
        )

    def request_finish(self, reason: str = "") -> None:
        self.finish_requested = True
        self.finish_reason = str(reason)
        gate = self.activation_dag.gates.get("gate:finish_eligible")
        if gate is not None:
            self.activation_dag.replace_node(
                GateNode(
                    gate_id=gate.gate_id,
                    condition={"op": "true"},
                    satisfied=True,
                    metadata={"reason": self.finish_reason},
                )
            )

    def record_completion_audit(self, passed: bool) -> None:
        """Update the fixed Audit/Render nodes after a mechanical audit."""

        audit = self.activation_dag.audits.get("audit:completion")
        if audit is not None:
            self.activation_dag.replace_node(
                AuditNode(
                    audit_id=audit.audit_id,
                    policy=dict(audit.policy),
                    state="completed" if passed else "failed",
                )
            )
        render = self.activation_dag.renders.get("render:final")
        if render is not None:
            self.activation_dag.replace_node(
                RenderNode(
                    render_id=render.render_id,
                    answer_kind=render.answer_kind,
                    state="ready" if passed else "dormant",
                )
            )

    def summary(self) -> dict[str, Any]:
        """Return a bounded graph summary suitable for a planner message."""

        return {
            "graph_version": self.version,
            "plan_version": self.plan_version,
            "contract": {
                "answer_kind": self.contract.answer_kind,
                "output_columns": self.contract.output_columns,
            }
            if self.contract
            else None,
            "frontier": {
                "ready": sorted(
                    action_id
                    for action_id, action in self.activation_dag.actions.items()
                    if action.state == ActionState.READY
                ),
                "running": sorted(
                    action_id
                    for action_id, action in self.activation_dag.actions.items()
                    if action.state == ActionState.RUNNING
                ),
                "blocked": sorted(
                    action_id
                    for action_id, action in self.activation_dag.actions.items()
                    if action.state == ActionState.BLOCKED
                ),
            },
            "evidence_counts": {
                kind.value: len(self.evidence_graph.iter_kind(kind))
                for kind in EvidenceKind
            },
            "open_conflicts": [
                node.node_id
                for node in self.evidence_graph.iter_kind(EvidenceKind.CONFLICT)
                if node.status in {EvidenceStatus.OPEN, EvidenceStatus.CONFLICTED}
            ],
            "recent_events": sum(len(queue) for queue in self.event_queues.values()),
        }

    def snapshot(self, *, max_nodes: int | None = None) -> dict[str, Any]:
        """Return a bounded, ground-truth-free debug snapshot."""

        limit = max_nodes or self.config.max_snapshot_nodes
        nodes = list(self.evidence_graph.nodes.values())[:limit]
        return {
            "graph_version": self.version,
            "plan_version": self.plan_version,
            "question": self.question,
            "answer_type": self.answer_type,
            "language": self.language,
            "format_requirements": copy.deepcopy(self.format_requirements),
            "budget": self.config.max_actions if self.budget is None else self.budget,
            "remaining_budget": self.remaining_budget,
            "contract": self.contract,
            "evidence_nodes": nodes,
            "evidence_edges": list(self.evidence_graph.edges.values())[: limit * 2],
            "activation": self.summary()["frontier"],
        }

    def context_token(self):
        """Set this runtime as the current ContextVar value."""

        return _graph_runtime_ctx.set(self)

    @staticmethod
    def reset_context(token) -> None:
        _graph_runtime_ctx.reset(token)

    def action_context_token(self, action_id: str | None):
        """Set the current action ContextVar and return its reset token."""

        return _current_action_ctx.set(action_id)

    @staticmethod
    def reset_action_context(token) -> None:
        _current_action_ctx.reset(token)

    @staticmethod
    def sub_traj_context_token(sub_traj_id: int):
        """Set the current sub-trajectory ContextVar."""

        return _current_sub_traj_ctx.set(int(sub_traj_id))

    @staticmethod
    def reset_sub_traj_context(token) -> None:
        _current_sub_traj_ctx.reset(token)
