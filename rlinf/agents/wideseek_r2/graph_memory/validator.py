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

"""Validation and atomic canonical commit for graph-memory proposals."""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.conditions import compile_condition
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionState,
    ActivationEdge,
    ActivationEdgeType,
    CommitResult,
    CompletenessType,
    EvidenceEdge,
    EvidenceKind,
    EvidenceNode,
    EvidenceProposal,
    EvidenceStatus,
    GraphDelta,
    GraphEventType,
    NodeProposal,
    PayloadNode,
    TaskContract,
    TaskPlanProposal,
)
from rlinf.agents.wideseek_r2.graph_memory.state import (
    EvidenceGraph,
    GraphRuntime,
    GraphStateError,
)


class GraphValidationError(ValueError):
    """Structured validation failure with a stable error code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


_ALLOWED_EVIDENCE_RELATIONS = {
    "OBSERVED_IN",
    "ABOUT",
    "NORMALIZES_TO",
    "SUPPORTS",
    "REFUTES",
    "VERIFIED_AS",
    "CONTRADICTS",
    "SAME_AS",
    "DERIVED_FROM",
    "MEMBER_OF",
    "HAS_FIELD",
    "PRECEDES",
}

_EVIDENCE_EDGE_KINDS: dict[str, tuple[set[EvidenceKind], set[EvidenceKind]]] = {
    "OBSERVED_IN": ({EvidenceKind.CANDIDATE}, {EvidenceKind.SOURCE}),
    "ABOUT": (
        {
            EvidenceKind.SOURCE,
            EvidenceKind.CANDIDATE,
            EvidenceKind.CLAIM,
            EvidenceKind.FACT,
            EvidenceKind.ENTITY,
        },
        {EvidenceKind.ENTITY},
    ),
    "NORMALIZES_TO": (
        {EvidenceKind.CANDIDATE},
        {EvidenceKind.ENTITY, EvidenceKind.CLAIM},
    ),
    "SUPPORTS": (
        {EvidenceKind.SOURCE, EvidenceKind.CANDIDATE, EvidenceKind.CLAIM},
        {EvidenceKind.CLAIM},
    ),
    "REFUTES": (
        {EvidenceKind.SOURCE, EvidenceKind.CANDIDATE, EvidenceKind.CLAIM},
        {EvidenceKind.CLAIM},
    ),
    "VERIFIED_AS": ({EvidenceKind.CLAIM}, {EvidenceKind.FACT}),
    "CONTRADICTS": (
        {EvidenceKind.CLAIM, EvidenceKind.FACT},
        {EvidenceKind.CLAIM, EvidenceKind.FACT, EvidenceKind.CONFLICT},
    ),
    "SAME_AS": ({EvidenceKind.ENTITY}, {EvidenceKind.ENTITY}),
    "DERIVED_FROM": (
        {EvidenceKind.CLAIM, EvidenceKind.FACT},
        {EvidenceKind.CANDIDATE, EvidenceKind.CLAIM, EvidenceKind.FACT},
    ),
    "MEMBER_OF": ({EvidenceKind.ENTITY}, {EvidenceKind.ENTITY}),
    "HAS_FIELD": ({EvidenceKind.ENTITY}, {EvidenceKind.FACT}),
    "PRECEDES": (
        {EvidenceKind.ENTITY, EvidenceKind.FACT},
        {EvidenceKind.ENTITY, EvidenceKind.FACT},
    ),
}


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _coerce_evidence_kind(node: Any) -> EvidenceKind:
    try:
        return node.normalized_kind()
    except ValueError as exc:
        raise GraphValidationError("INVALID_EVIDENCE_NODE", str(exc)) from exc


def _remap_value(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_remap_value(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_value(item, mapping) for item in value)
    if isinstance(value, dict):
        return {key: _remap_value(item, mapping) for key, item in value.items()}
    return value


class GraphValidator:
    """Schema, provenance, ACL, version, and DAG validator."""

    def validate_task_plan(
        self, runtime: GraphRuntime, proposal: TaskPlanProposal
    ) -> TaskPlanProposal:
        try:
            contract = (
                proposal.contract
                if isinstance(proposal.contract, TaskContract)
                else TaskContract.from_dict(
                    proposal.contract, question=runtime.question
                )
            )
        except (TypeError, ValueError) as exc:
            raise GraphValidationError("INVALID_CONTRACT", str(exc)) from exc
        if contract.question != runtime.question:
            raise GraphValidationError(
                "CONTRACT_QUESTION_MISMATCH",
                "Task Contract question must match the rollout question",
            )
        if runtime.answer_type == "item" and contract.answer_kind != "item":
            raise GraphValidationError(
                "ANSWER_TYPE_MISMATCH",
                "Phase 1 mas_graph only supports answer_type=item",
            )
        if contract.completeness_policy.type not in {
            item.value for item in CompletenessType
        }:
            raise GraphValidationError(
                "INVALID_COMPLETENESS_POLICY",
                f"Unknown completion policy {contract.completeness_policy.type!r}",
            )
        if (
            contract.answer_kind == "item"
            and contract.completeness_policy.type
            != CompletenessType.DEPENDENCY_TERMINAL.value
        ):
            raise GraphValidationError(
                "UNSUPPORTED_COMPLETENESS_POLICY",
                "Phase 1 item plans require dependency_terminal completeness",
            )

        action_ids = [action.action_id for action in proposal.actions]
        gate_ids = [gate.gate_id for gate in proposal.gates]
        payload_ids = [payload.payload_id for payload in proposal.payloads]
        join_ids = [join.join_id for join in proposal.joins]
        audit_ids = [audit.audit_id for audit in proposal.audits]
        render_ids = [render.render_id for render in proposal.renders]
        for label, values in (
            ("action", action_ids),
            ("gate", gate_ids),
            ("payload", payload_ids),
            ("join", join_ids),
            ("audit", audit_ids),
            ("render", render_ids),
        ):
            if any(not value for value in values):
                raise GraphValidationError(
                    "INVALID_NODE_ID", f"{label} IDs cannot be empty"
                )
            if len(values) != len(set(values)):
                raise GraphValidationError("DUPLICATE_NODE_ID", f"Duplicate {label} ID")
        if any(action.owner_sub_traj is not None for action in proposal.actions):
            raise GraphValidationError(
                "ACTION_OWNER_FORBIDDEN",
                "Action ownership is assigned by the scheduler, not the planner",
            )
        all_proposed_ids = (
            action_ids + gate_ids + payload_ids + join_ids + audit_ids + render_ids
        )
        if len(all_proposed_ids) != len(set(all_proposed_ids)):
            raise GraphValidationError(
                "DUPLICATE_NODE_ID", "Activation node IDs must be globally unique"
            )
        edge_ids = [edge.edge_id for edge in proposal.edges]
        if any(not edge_id for edge_id in edge_ids):
            raise GraphValidationError("INVALID_EDGE_ID", "Edge IDs cannot be empty")
        if len(edge_ids) != len(set(edge_ids)):
            raise GraphValidationError("DUPLICATE_EDGE_ID", "Duplicate edge ID")
        edge_collisions = sorted(set(edge_ids) & set(runtime.activation_dag.edges))
        if edge_collisions:
            raise GraphValidationError(
                "RESERVED_EDGE_ID",
                f"Plan cannot replace bootstrap activation edges: {edge_collisions}",
            )
        reserved_ids = set(runtime.activation_dag.nodes)
        collisions = sorted(set(all_proposed_ids) & reserved_ids)
        if collisions:
            raise GraphValidationError(
                "RESERVED_NODE_ID",
                f"Plan cannot replace bootstrap activation nodes: {collisions}",
            )
        system_action_ids = {"action:plan_task", "action:initial_frontier"}
        existing_user_action_count = sum(
            action_id not in system_action_ids
            for action_id in runtime.activation_dag.actions
        )
        if existing_user_action_count + len(action_ids) > runtime.config.max_actions:
            raise GraphValidationError(
                "ACTION_LIMIT", "Activation action budget exceeded"
            )
        for gate in proposal.gates:
            try:
                condition = compile_condition(gate.condition)
            except Exception as exc:
                raise GraphValidationError("INVALID_GATE", str(exc)) from exc
            if condition != gate.condition:
                # A normalized condition is safer than accepting a subtly
                # different structure, and retains the original proposal API.
                pass

        anchor_refs = {anchor.canonical_key for anchor in proposal.anchor_entities}
        normalized_payloads = []
        for payload in proposal.payloads:
            selector = payload.selector or {}
            if payload.max_tokens > runtime.config.max_notification_tokens:
                payload = PayloadNode(
                    **{
                        **payload.__dict__,
                        "max_tokens": runtime.config.max_notification_tokens,
                    }
                )
            normalized_payloads.append(payload)
            try:
                delivery_policy = str(payload.delivery_policy)
                if delivery_policy.startswith("DeliveryPolicy."):
                    delivery_policy = delivery_policy.rsplit(".", 1)[-1].lower()
                if delivery_policy not in {"on_ready", "on_change", "manual_pull"}:
                    raise ValueError(
                        f"unsupported delivery policy {payload.delivery_policy!r}"
                    )
            except (TypeError, ValueError) as exc:
                raise GraphValidationError("INVALID_PAYLOAD", str(exc)) from exc
            explicit_refs = selector.get("refs", [])
            if isinstance(explicit_refs, str):
                explicit_refs = [explicit_refs]
            if explicit_refs:
                for ref in explicit_refs:
                    if (
                        str(ref) not in runtime.evidence_graph.nodes
                        and runtime.evidence_graph.get_by_canonical(str(ref)) is None
                    ):
                        raise GraphValidationError(
                            "UNKNOWN_EVIDENCE_REF",
                            f"Payload {payload.payload_id!r} references unknown evidence {ref!r}",
                        )
            gate_ref = selector.get("refs_from_gate")
            known_gate_ids = set(runtime.activation_dag.gates) | set(gate_ids)
            if gate_ref and str(gate_ref) not in known_gate_ids:
                raise GraphValidationError(
                    "UNKNOWN_ACTIVATION_REF",
                    f"Payload {payload.payload_id!r} references unknown gate {gate_ref!r}",
                )
            entity_ref = selector.get("entity_ref")
            if entity_ref and str(entity_ref) not in anchor_refs:
                entity = runtime.evidence_graph.nodes.get(str(entity_ref))
                if (
                    entity is None
                    and runtime.evidence_graph.get_by_canonical(
                        str(entity_ref), EvidenceKind.ENTITY
                    )
                    is None
                ):
                    # Per-entity selectors may be filled by a later dynamic
                    # expansion; explicit unknown refs are not silently read.
                    raise GraphValidationError(
                        "UNKNOWN_EVIDENCE_REF",
                        f"Payload {payload.payload_id!r} references unknown entity {entity_ref!r}",
                    )

        known = {
            "action:plan_task",
            "action:initial_frontier",
            "gate:contract_valid",
            "gate:finish_eligible",
            "audit:completion",
            "render:final",
            "payload:task_context",
        }
        known.update(runtime.activation_dag.nodes)
        known.update(action_ids)
        known.update(gate_ids)
        known.update(payload_ids)
        known.update(join.join_id for join in proposal.joins)
        known.update(audit.audit_id for audit in proposal.audits)
        known.update(render.render_id for render in proposal.renders)
        for action in proposal.actions:
            for ref in action.predecessor_ids + action.guard_ids + action.payload_ids:
                if ref not in known:
                    raise GraphValidationError(
                        "UNKNOWN_ACTIVATION_REF",
                        f"Action {action.action_id!r} references unknown node {ref!r}",
                    )
        for edge in proposal.edges:
            if edge.source_id not in known or edge.target_id not in known:
                raise GraphValidationError(
                    "UNKNOWN_ACTIVATION_REF",
                    f"Unknown activation edge endpoint {edge.source_id!r} -> {edge.target_id!r}",
                )
            try:
                ActivationEdgeType(
                    edge.relation
                    if isinstance(edge.relation, ActivationEdgeType)
                    else str(edge.relation)
                )
            except ValueError as exc:
                raise GraphValidationError(
                    "INVALID_ACTIVATION_EDGE", str(edge.relation)
                ) from exc

        # Install into a temporary DAG so cycle detection happens before any
        # canonical state is changed.
        from rlinf.agents.wideseek_r2.graph_memory.state import ActivationDAG

        temp = ActivationDAG()
        for node in runtime.activation_dag.nodes.values():
            temp.add_node(node[1])
        for node in (
            proposal.payloads + proposal.gates + proposal.joins + proposal.actions
        ):
            if temp.node_id(node) not in temp.nodes:
                temp.add_node(node)
        for node in proposal.audits + proposal.renders:
            if temp.node_id(node) not in temp.nodes:
                temp.add_node(node)
        for edge in runtime.activation_dag.edges.values():
            temp.add_edge(edge)
        for edge in proposal.edges:
            if edge.edge_id not in temp.edges:
                temp.add_edge(edge)
        for action in proposal.actions:
            for index, predecessor_id in enumerate(action.predecessor_ids):
                temp.add_edge(
                    ActivationEdge(
                        edge_id=f"__field_precedes:{action.action_id}:{index}",
                        source_id=predecessor_id,
                        target_id=action.action_id,
                        relation=ActivationEdgeType.PRECEDES,
                    )
                )
            for index, guard_id in enumerate(action.guard_ids):
                temp.add_edge(
                    ActivationEdge(
                        edge_id=f"__field_guards:{action.action_id}:{index}",
                        source_id=guard_id,
                        target_id=action.action_id,
                        relation=ActivationEdgeType.GUARDS,
                    )
                )
            for index, payload_id in enumerate(action.payload_ids):
                temp.add_edge(
                    ActivationEdge(
                        edge_id=f"__field_delivers:{action.action_id}:{index}",
                        source_id=payload_id,
                        target_id=action.action_id,
                        relation=ActivationEdgeType.DELIVERS,
                    )
                )
        try:
            temp.validate_acyclic()
        except GraphStateError as exc:
            raise GraphValidationError("DAG_CYCLE", str(exc)) from exc
        anchor_keys: set[str] = set()
        for anchor in proposal.anchor_entities:
            try:
                kind = anchor.normalized_kind()
            except ValueError as exc:
                raise GraphValidationError("INVALID_ANCHOR_KIND", str(exc)) from exc
            if kind != EvidenceKind.ENTITY:
                raise GraphValidationError(
                    "INVALID_ANCHOR_KIND",
                    f"Plan anchors must be Entity nodes, got {kind.value}",
                )
            if not anchor.canonical_key:
                raise GraphValidationError(
                    "INVALID_CANONICAL_KEY", "Anchor canonical_key is required"
                )
            if anchor.canonical_key in anchor_keys:
                raise GraphValidationError(
                    "DUPLICATE_ANCHOR",
                    f"Duplicate anchor canonical key {anchor.canonical_key!r}",
                )
            anchor_keys.add(anchor.canonical_key)
        return TaskPlanProposal(
            contract=contract,
            actions=tuple(
                action
                if action.state == ActionState.DORMANT
                else type(action)(**{**action.__dict__, "state": ActionState.DORMANT})
                for action in proposal.actions
            ),
            gates=proposal.gates,
            payloads=tuple(normalized_payloads),
            joins=proposal.joins,
            audits=proposal.audits,
            renders=proposal.renders,
            edges=proposal.edges,
            anchor_entities=proposal.anchor_entities,
            proposal_id=proposal.proposal_id,
        )

    def validate_evidence_proposal(
        self,
        runtime: GraphRuntime,
        proposal: EvidenceProposal,
        *,
        action_id: str | None = None,
    ) -> None:
        effective_action = action_id or proposal.action_id
        action = runtime.activation_dag.actions.get(effective_action)
        if action is None:
            raise GraphValidationError(
                "UNKNOWN_ACTION", f"Unknown action {effective_action!r}"
            )
        if action.state not in {ActionState.READY, ActionState.RUNNING}:
            raise GraphValidationError(
                "ACTION_NOT_RUNNING",
                f"Action {effective_action!r} is in state {action.state.value!r}",
            )
        if (
            action.owner_sub_traj is not None
            and proposal.created_by_sub_traj != action.owner_sub_traj
        ):
            raise GraphValidationError(
                "ACTION_OWNER_MISMATCH",
                f"Action {effective_action!r} is owned by sub-trajectory "
                f"{action.owner_sub_traj}, proposal came from {proposal.created_by_sub_traj}",
            )
        if not proposal.nodes and not proposal.action_result.summary:
            raise GraphValidationError(
                "EMPTY_PROPOSAL", "Evidence proposal must contain nodes or a summary"
            )
        if proposal.action_result.status.lower() not in {
            "completed",
            "failed",
            "blocked",
            "invalidated",
            "running",
        }:
            raise GraphValidationError(
                "INVALID_ACTION_STATUS",
                f"Unsupported action status {proposal.action_result.status!r}",
            )
        if proposal.base_version > runtime.version:
            raise GraphValidationError(
                "VERSION_CONFLICT",
                f"Proposal base version {proposal.base_version} is newer than graph version {runtime.version}",
            )
        if proposal.base_version < runtime.version:
            # A stale proposal is safe only if it contains no new canonical
            # keys; callers can retry with the current version otherwise.
            stale_new = [
                node
                for node in proposal.nodes
                if runtime.evidence_graph.get_by_canonical(
                    node.canonical_key,
                    _coerce_evidence_kind(node),
                )
                is None
            ]
            if stale_new:
                raise GraphValidationError(
                    "VERSION_CONFLICT",
                    f"Proposal is stale at {proposal.base_version}; current graph version is {runtime.version}",
                )
        refs = set()
        for node in proposal.nodes:
            if not node.client_ref or node.client_ref in refs:
                raise GraphValidationError(
                    "DUPLICATE_CLIENT_REF",
                    f"Duplicate/empty client ref {node.client_ref!r}",
                )
            refs.add(node.client_ref)
            if not node.canonical_key:
                raise GraphValidationError(
                    "INVALID_CANONICAL_KEY", "canonical_key is required"
                )
            try:
                kind = node.normalized_kind()
                node.normalized_status()
            except ValueError as exc:
                raise GraphValidationError("INVALID_EVIDENCE_NODE", str(exc)) from exc
            if (
                kind == EvidenceKind.FACT
                and not runtime.config.allow_worker_fact_proposal
            ):
                raise GraphValidationError(
                    "WORKER_FACT_FORBIDDEN",
                    "Workers may propose Claims; the system creates verified Facts",
                )
            if kind == EvidenceKind.SOURCE:
                uri = node.payload.get("uri")
                if not isinstance(uri, str) or not uri.strip():
                    raise GraphValidationError(
                        "SOURCE_PROVENANCE_REQUIRED",
                        "Source payload requires a non-empty uri",
                    )
                locator = node.payload.get("locator")
                if locator is None or not str(locator).strip():
                    raise GraphValidationError(
                        "SOURCE_LOCATOR_REQUIRED",
                        "Source payload requires a non-empty locator",
                    )
            if kind == EvidenceKind.CLAIM:
                if not node.payload.get("predicate"):
                    raise GraphValidationError(
                        "CLAIM_PREDICATE_REQUIRED", "Claim payload requires predicate"
                    )
                subject_ref = node.payload.get("subject_ref")
                if not subject_ref:
                    raise GraphValidationError(
                        "CLAIM_SUBJECT_REQUIRED", "Claim payload requires subject_ref"
                    )
                if (
                    subject_ref
                    and subject_ref not in refs
                    and subject_ref not in runtime.evidence_graph.nodes
                    and runtime.evidence_graph.get_by_canonical(str(subject_ref))
                    is None
                ):
                    raise GraphValidationError(
                        "UNKNOWN_CLAIM_SUBJECT",
                        f"Claim subject ref {subject_ref!r} is not in this proposal or graph",
                    )
        existing_refs = set(runtime.evidence_graph.nodes)
        kind_by_ref = {
            node.client_ref: node.normalized_kind() for node in proposal.nodes
        }

        def resolve_kind(ref: str) -> EvidenceKind | None:
            if ref in kind_by_ref:
                return kind_by_ref[ref]
            existing = runtime.evidence_graph.nodes.get(
                ref
            ) or runtime.evidence_graph.get_by_canonical(str(ref))
            return existing.kind if existing is not None else None

        for edge in proposal.edges:
            relation = str(edge.relation).upper()
            if relation not in _ALLOWED_EVIDENCE_RELATIONS:
                raise GraphValidationError(
                    "INVALID_EVIDENCE_EDGE",
                    f"Unknown evidence relation {edge.relation!r}",
                )
            for ref in (edge.source_ref, edge.target_ref):
                if (
                    ref not in refs
                    and ref not in existing_refs
                    and runtime.evidence_graph.get_by_canonical(str(ref)) is None
                ):
                    raise GraphValidationError(
                        "UNKNOWN_EVIDENCE_REF", f"Unknown evidence ref {ref!r}"
                    )
            endpoint_kinds = _EVIDENCE_EDGE_KINDS.get(relation)
            if endpoint_kinds is not None:
                source_kind = resolve_kind(edge.source_ref)
                target_kind = resolve_kind(edge.target_ref)
                if (
                    source_kind is not None
                    and target_kind is not None
                    and (
                        source_kind not in endpoint_kinds[0]
                        or target_kind not in endpoint_kinds[1]
                    )
                ):
                    raise GraphValidationError(
                        "INVALID_EVIDENCE_EDGE_ENDPOINTS",
                        f"{relation} does not allow {source_kind.value} -> {target_kind.value}",
                    )

    @staticmethod
    def _resolve_ref(ref: str, mapping: dict[str, str], graph: EvidenceGraph) -> str:
        resolved = mapping.get(ref, ref)
        if resolved not in graph.nodes:
            canonical = graph.get_by_canonical(resolved)
            if canonical is not None:
                resolved = canonical.node_id
        if resolved not in graph.nodes:
            raise GraphValidationError(
                "UNKNOWN_EVIDENCE_REF", f"Unknown evidence ref {ref!r}"
            )
        return resolved

    def _canonicalize_nodes(
        self, runtime: GraphRuntime, proposal: EvidenceProposal
    ) -> tuple[dict[str, str], list[str], list[str]]:
        graph = runtime.evidence_graph
        mapping: dict[str, str] = {}
        changed_node_ids: list[str] = []
        new_claim_ids: list[str] = []
        for node_proposal in proposal.nodes:
            kind = node_proposal.normalized_kind()
            existing = graph.get_by_canonical(node_proposal.canonical_key, kind)
            if existing is not None:
                mapping[node_proposal.client_ref] = existing.node_id
                if kind == EvidenceKind.CLAIM:
                    new_claim_ids.append(existing.node_id)
                continue
            node_id = _stable_id(f"evidence:{kind.value}", node_proposal.canonical_key)
            node = EvidenceNode(
                node_id=node_id,
                kind=kind,
                canonical_key=node_proposal.canonical_key,
                payload={},
                status=node_proposal.normalized_status(),
                created_by_action=proposal.action_id,
                created_by_sub_traj=proposal.created_by_sub_traj,
                created_at_version=runtime.version + 1,
                confidence=node_proposal.confidence,
                tags=tuple(node_proposal.tags),
            )
            graph.add_node(node)
            mapping[node_proposal.client_ref] = node_id
            changed_node_ids.append(node_id)
            if kind == EvidenceKind.CLAIM:
                new_claim_ids.append(node_id)
        # Payload references can point to another node in the same proposal.
        for node_proposal in proposal.nodes:
            node_id = mapping[node_proposal.client_ref]
            node = graph.nodes[node_id]
            payload = _remap_value(copy.deepcopy(node_proposal.payload), mapping)
            if node.kind == EvidenceKind.CLAIM:
                subject_ref = payload.get("subject_ref")
                if subject_ref:
                    subject = graph.nodes.get(
                        str(subject_ref)
                    ) or graph.get_by_canonical(str(subject_ref))
                    if subject is not None:
                        payload["subject_ref"] = subject.node_id
            if payload:
                graph.replace_node(
                    EvidenceNode(
                        **{
                            **node.__dict__,
                            "payload": {**node.payload, **payload},
                        }
                    )
                )
        return mapping, changed_node_ids, new_claim_ids

    def _commit_locked(
        self, runtime: GraphRuntime, proposal: EvidenceProposal
    ) -> CommitResult:
        self.validate_evidence_proposal(runtime, proposal)
        if runtime.version >= runtime.config.max_graph_versions:
            raise GraphValidationError(
                "GRAPH_VERSION_LIMIT", "graph version budget exceeded"
            )

        # The canonical graph and activation state are updated together. Keep
        # a private rollback image so a late limit/selector error cannot leave
        # half of a proposal visible to another worker.
        rollback = {
            "evidence_graph": copy.deepcopy(runtime.evidence_graph),
            "activation_dag": copy.deepcopy(runtime.activation_dag),
            "version": runtime.version,
            "remaining_budget": runtime.remaining_budget,
            "event_queues": copy.deepcopy(runtime.event_queues),
            "emitted_event_keys": set(runtime.emitted_event_keys),
            "events_by_key": copy.deepcopy(runtime.events_by_key),
            "delivered_event_keys": set(runtime.delivered_event_keys),
            "action_acl": copy.deepcopy(runtime.action_acl),
            "action_acl_expiry": dict(runtime.action_acl_expiry),
            "action_results": copy.deepcopy(runtime.action_results),
            "covered_partitions": set(runtime.covered_partitions),
            "last_error": runtime.last_error,
        }
        graph = runtime.evidence_graph
        try:
            mapping, changed_node_ids, claim_ids = self._canonicalize_nodes(
                runtime, proposal
            )
            changed_edge_ids: list[str] = []
            for edge_proposal in proposal.edges:
                source_id = self._resolve_ref(edge_proposal.source_ref, mapping, graph)
                target_id = self._resolve_ref(edge_proposal.target_ref, mapping, graph)
                relation = str(edge_proposal.relation).upper()
                edge_id = _stable_id(
                    "edge",
                    f"{source_id}|{relation}|{target_id}",
                )
                edge, inserted = graph.add_edge(
                    EvidenceEdge(
                        edge_id=edge_id,
                        source_id=source_id,
                        relation=relation,
                        target_id=target_id,
                        created_by_action=proposal.action_id,
                        created_at_version=runtime.version + 1,
                    )
                )
                if inserted:
                    changed_edge_ids.append(edge.edge_id)
                if relation in {"SUPPORTS", "REFUTES", "ABOUT", "DERIVED_FROM"}:
                    if (
                        graph.nodes.get(target_id, None) is not None
                        and graph.nodes[target_id].kind == EvidenceKind.CLAIM
                    ):
                        claim_ids.append(target_id)
                    if (
                        graph.nodes.get(source_id, None) is not None
                        and graph.nodes[source_id].kind == EvidenceKind.CLAIM
                    ):
                        claim_ids.append(source_id)

            fact_ids: list[str] = []
            for claim_id in set(claim_ids):
                fact = self._maybe_accept_claim(runtime, claim_id, proposal.action_id)
                if fact is not None:
                    fact_ids.append(fact.node_id)
            conflict_ids = self._detect_conflicts(
                runtime, claim_ids, proposal.action_id
            )
            if len(graph.nodes) > runtime.config.max_nodes:
                raise GraphValidationError(
                    "NODE_LIMIT", "Evidence node budget exceeded"
                )
            if len(graph.edges) > runtime.config.max_edges:
                raise GraphValidationError(
                    "EDGE_LIMIT", "Evidence edge budget exceeded"
                )
            runtime.version += 1
            graph.version = runtime.version

            status = proposal.action_result.status.lower()
            if status in {"completed", "failed", "blocked", "invalidated"}:
                runtime.mark_action_completed(
                    proposal.action_id,
                    status=status,
                    summary=proposal.action_result.summary,
                    partition=proposal.action_result.partition_cursor,
                )
            runtime.action_results.setdefault(proposal.action_id, {}).update(
                {
                    "proposal_id": proposal.proposal_id,
                    "accepted_node_count": len(changed_node_ids),
                    "accepted_edge_count": len(changed_edge_ids),
                    "accepted_fact_count": len(fact_ids),
                }
            )
            transitions = runtime.evaluate_activation()
            return CommitResult(
                proposal_id=proposal.proposal_id,
                graph_version=runtime.version,
                delta=GraphDelta(
                    node_ids=tuple(changed_node_ids),
                    edge_ids=tuple(changed_edge_ids),
                    fact_ids=tuple(fact_ids),
                    conflict_ids=tuple(conflict_ids),
                ),
                transitions=tuple(transitions),
            )
        except Exception:
            for field_name, value in rollback.items():
                setattr(runtime, field_name, value)
            raise

    def _maybe_accept_claim(
        self, runtime: GraphRuntime, claim_id: str, action_id: str
    ) -> EvidenceNode | None:
        graph = runtime.evidence_graph
        claim = graph.nodes.get(claim_id)
        if claim is None or claim.kind != EvidenceKind.CLAIM:
            return None
        source_ids: set[str] = set()
        pending = [claim_id]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            for edge in graph.incoming(current):
                source = graph.nodes.get(edge.source_id)
                if source is None:
                    continue
                if source.kind == EvidenceKind.SOURCE:
                    source_ids.add(source.node_id)
                elif edge.relation in {
                    "SUPPORTS",
                    "OBSERVED_IN",
                    "NORMALIZES_TO",
                    "DERIVED_FROM",
                }:
                    pending.append(source.node_id)
            for edge in graph.outgoing(current):
                if edge.relation != "OBSERVED_IN":
                    continue
                source = graph.nodes.get(edge.target_id)
                if source is not None and source.kind == EvidenceKind.SOURCE:
                    source_ids.add(source.node_id)
        policy = runtime.config
        if (
            policy.require_source_for_verified_claim
            and len(source_ids) < policy.min_independent_sources
        ):
            if claim.status != EvidenceStatus.SUPPORTED:
                graph.replace_node(claim.with_status(EvidenceStatus.SUPPORTED))
            return None
        fact_key = f"{claim.canonical_key}:verified"
        existing = graph.get_by_canonical(fact_key, EvidenceKind.FACT)
        if existing is not None:
            return existing
        fact_payload = {
            "claim_ref": claim.node_id,
            "subject_ref": claim.payload.get("subject_ref"),
            "predicate": claim.payload.get("predicate"),
            "object": claim.payload.get("object"),
            "value": claim.payload.get("value", claim.payload.get("object")),
        }
        fact_tags = list(claim.tags)
        terminal_tag = runtime.config.item_terminal_tag
        if (
            claim.payload.get("terminal")
            or "terminal" in fact_tags
            or terminal_tag in fact_tags
        ):
            fact_tags.append(terminal_tag)
        fact = EvidenceNode(
            node_id=_stable_id("evidence:fact", fact_key),
            kind=EvidenceKind.FACT,
            canonical_key=fact_key,
            payload=fact_payload,
            status=EvidenceStatus.VERIFIED,
            created_by_action=action_id,
            created_by_sub_traj=claim.created_by_sub_traj,
            created_at_version=runtime.version + 1,
            confidence=claim.confidence,
            tags=tuple(dict.fromkeys(fact_tags)),
        )
        graph.add_node(fact)
        graph.add_edge(
            EvidenceEdge(
                edge_id=_stable_id(
                    "edge", f"{claim.node_id}|VERIFIED_AS|{fact.node_id}"
                ),
                source_id=claim.node_id,
                relation="VERIFIED_AS",
                target_id=fact.node_id,
                created_by_action=action_id,
                created_at_version=runtime.version + 1,
            )
        )
        graph.replace_node(claim.with_status(EvidenceStatus.VERIFIED))
        return fact

    def _detect_conflicts(
        self, runtime: GraphRuntime, claim_ids: list[str], action_id: str
    ) -> list[str]:
        graph = runtime.evidence_graph
        groups: dict[tuple[Any, Any], list[EvidenceNode]] = defaultdict(list)
        for claim in graph.iter_kind(EvidenceKind.CLAIM):
            subject = claim.payload.get("subject_ref")
            predicate = claim.payload.get("predicate")
            if subject is not None and predicate is not None:
                groups[(subject, predicate)].append(claim)
        conflict_ids: list[str] = []
        for (subject, predicate), claims in groups.items():
            values = {
                repr(claim.payload.get("object", claim.payload.get("value")))
                for claim in claims
            }
            if len(values) <= 1:
                continue
            conflict_key = f"{subject}:{predicate}:conflict"
            conflict = graph.get_by_canonical(conflict_key, EvidenceKind.CONFLICT)
            if conflict is None:
                conflict = EvidenceNode(
                    node_id=_stable_id("evidence:conflict", conflict_key),
                    kind=EvidenceKind.CONFLICT,
                    canonical_key=conflict_key,
                    payload={
                        "competing_refs": [claim.node_id for claim in claims],
                        "conflict_type": "incompatible_claim_values",
                        "resolution_policy": "verification_action",
                    },
                    status=EvidenceStatus.OPEN,
                    created_by_action=action_id,
                    created_at_version=runtime.version + 1,
                    tags=("open_conflict",),
                )
                graph.add_node(conflict)
            else:
                graph.replace_node(
                    EvidenceNode(
                        **{
                            **conflict.__dict__,
                            "payload": {
                                **conflict.payload,
                                "competing_refs": [claim.node_id for claim in claims],
                            },
                            "status": EvidenceStatus.OPEN,
                        }
                    )
                )
            conflict_ids.append(conflict.node_id)
            for claim in claims:
                if claim.status != EvidenceStatus.CONFLICTED:
                    graph.replace_node(claim.with_status(EvidenceStatus.CONFLICTED))
                for fact in graph.facts_for_claim(claim.node_id):
                    graph.replace_node(
                        EvidenceNode(
                            **{
                                **fact.with_status(EvidenceStatus.INVALIDATED).__dict__,
                                "payload": {
                                    **fact.payload,
                                    "invalidated_by": conflict.node_id,
                                },
                            }
                        )
                    )
                graph.add_edge(
                    EvidenceEdge(
                        edge_id=_stable_id(
                            "edge", f"{claim.node_id}|CONTRADICTS|{conflict.node_id}"
                        ),
                        source_id=claim.node_id,
                        relation="CONTRADICTS",
                        target_id=conflict.node_id,
                        created_by_action=action_id,
                        created_at_version=runtime.version + 1,
                    )
                )
        return list(dict.fromkeys(conflict_ids))


_V2_SUBAGENT_NODE_KINDS = {EvidenceKind.SOURCE, EvidenceKind.CANDIDATE}
_V2_MAIN_NODE_KINDS = {
    EvidenceKind.ENTITY,
    EvidenceKind.CLAIM,
    EvidenceKind.FACT,
    EvidenceKind.CONFLICT,
}

_V2_EDGE_KINDS: dict[str, tuple[set[EvidenceKind], set[EvidenceKind]]] = {
    "OBSERVED_IN": ({EvidenceKind.CANDIDATE}, {EvidenceKind.SOURCE}),
    "ABOUT": (
        {EvidenceKind.CANDIDATE, EvidenceKind.CLAIM, EvidenceKind.FACT},
        {EvidenceKind.ENTITY},
    ),
    "MENTIONS": ({EvidenceKind.SOURCE}, {EvidenceKind.ENTITY}),
    "NORMALIZES_TO": ({EvidenceKind.CANDIDATE}, {EvidenceKind.ENTITY}),
    "SAME_AS": ({EvidenceKind.ENTITY}, {EvidenceKind.ENTITY}),
    "SUPPORTED_BY": (
        {EvidenceKind.CLAIM, EvidenceKind.FACT},
        {EvidenceKind.CANDIDATE, EvidenceKind.SOURCE, EvidenceKind.FACT},
    ),
    "VERIFIED_AS": ({EvidenceKind.CLAIM}, {EvidenceKind.FACT}),
    "CONTAINS": (
        {EvidenceKind.CONFLICT},
        {EvidenceKind.CLAIM, EvidenceKind.FACT},
    ),
    "SUPERSEDES": ({EvidenceKind.FACT}, {EvidenceKind.FACT}),
}


def _v2_resolve_ref(
    runtime: GraphRuntime,
    ref: str,
    mapping: dict[str, str] | None = None,
    *,
    active_only: bool = True,
) -> str:
    """Resolve a client ref, node ID, or unambiguous canonical key."""

    graph = runtime.evidence_graph
    resolved = (mapping or {}).get(str(ref), str(ref))
    if resolved in graph.nodes:
        node = graph.nodes[resolved]
        if active_only and not node.active:
            raise GraphValidationError("INACTIVE_EVIDENCE_REF", str(ref))
        return resolved
    node = graph.get_by_canonical(resolved, active_only=active_only)
    if node is None:
        raise GraphValidationError(
            "UNKNOWN_EVIDENCE_REF", f"Unknown evidence ref {ref!r}"
        )
    return node.node_id


def _v2_snapshot(runtime: GraphRuntime) -> dict[str, Any]:
    """Capture every mutable field touched by a Phase 1 transaction."""

    return {
        "evidence_graph": copy.deepcopy(runtime.evidence_graph),
        "activation_dag": copy.deepcopy(runtime.activation_dag),
        "version": runtime.version,
        "event_log": copy.deepcopy(runtime.event_log),
        "pending_claim_ids": set(runtime.pending_claim_ids),
        "pending_conflict_ids": set(runtime.pending_conflict_ids),
        "action_results": copy.deepcopy(runtime.action_results),
        "bootstrap_entities": tuple(runtime.bootstrap_entities),
        "bootstrap_metadata": copy.deepcopy(runtime.bootstrap_metadata),
        "last_error": runtime.last_error,
    }


def _v2_restore(runtime: GraphRuntime, snapshot: dict[str, Any]) -> None:
    for name, value in snapshot.items():
        setattr(runtime, name, value)


def _v2_node_from_proposal(
    runtime: GraphRuntime,
    proposal: NodeProposal,
    *,
    actor_role: str,
    action_id: str,
    sub_traj_id: int,
    version: int,
) -> EvidenceNode:
    kind = proposal.normalized_kind()
    status = proposal.normalized_status()
    if status == EvidenceStatus.PROPOSED:
        status = {
            EvidenceKind.ENTITY: EvidenceStatus.ACTIVE,
            EvidenceKind.SOURCE: EvidenceStatus.ACTIVE,
            EvidenceKind.CANDIDATE: EvidenceStatus.ACTIVE,
            EvidenceKind.CLAIM: EvidenceStatus.PENDING,
            EvidenceKind.FACT: EvidenceStatus.ACTIVE,
            EvidenceKind.CONFLICT: EvidenceStatus.OPEN,
        }[kind]
    if kind == EvidenceKind.CLAIM and not proposal.payload.get("predicate"):
        raise GraphValidationError(
            "CLAIM_PREDICATE_REQUIRED", "Claim payload requires predicate"
        )
    if kind == EvidenceKind.CLAIM and not proposal.payload.get("subject_ref"):
        raise GraphValidationError(
            "CLAIM_SUBJECT_REQUIRED", "Claim payload requires subject_ref"
        )
    if kind == EvidenceKind.SOURCE:
        uri = proposal.payload.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            raise GraphValidationError(
                "SOURCE_PROVENANCE_REQUIRED",
                "Source payload requires a non-empty uri",
            )
        locator = proposal.payload.get("locator")
        if locator is None or not str(locator).strip():
            raise GraphValidationError(
                "SOURCE_LOCATOR_REQUIRED",
                "Source payload requires a non-empty locator",
            )
    return EvidenceNode(
        node_id=_stable_id(f"evidence:{kind.value}", proposal.canonical_key),
        kind=kind,
        canonical_key=proposal.canonical_key,
        payload=copy.deepcopy(proposal.payload),
        status=status,
        created_by_action=action_id,
        created_by_sub_traj=sub_traj_id,
        created_at_version=version,
        confidence=proposal.confidence,
        tags=tuple(proposal.tags),
        active=True,
        proposed_by_role=actor_role,  # type: ignore[arg-type]
        proposed_by_turn=runtime.main_turn,
        updated_at_version=version,
        tool_result_refs=tuple(proposal.tool_result_refs),
    )


def _v2_check_edge(
    runtime: GraphRuntime,
    relation: str,
    source_id: str,
    target_id: str,
    *,
    actor_role: str,
) -> None:
    relation = relation.upper()
    endpoint_kinds = _V2_EDGE_KINDS.get(relation)
    if endpoint_kinds is None:
        raise GraphValidationError(
            "INVALID_EVIDENCE_EDGE", f"Unknown Phase 1 relation {relation!r}"
        )
    source = runtime.evidence_graph.nodes[source_id]
    target = runtime.evidence_graph.nodes[target_id]
    if source.kind not in endpoint_kinds[0] or target.kind not in endpoint_kinds[1]:
        raise GraphValidationError(
            "INVALID_EVIDENCE_EDGE_ENDPOINTS",
            f"{relation} does not allow {source.kind.value} -> {target.kind.value}",
        )
    if actor_role == "subagent" and relation not in {
        "OBSERVED_IN",
        "ABOUT",
        "MENTIONS",
    }:
        raise GraphValidationError(
            "SUBAGENT_EDGE_FORBIDDEN",
            f"Subagents cannot create {relation} edges",
        )


def _v2_tool_refs(
    runtime: GraphRuntime,
    refs: tuple[str, ...] | list[str],
    *,
    action_id: str,
    sub_traj_id: int,
    node_kind: EvidenceKind,
) -> tuple[str, ...]:
    refs = tuple(dict.fromkeys(str(ref) for ref in refs if str(ref)))
    if runtime.config.require_tool_provenance and not refs:
        raise GraphValidationError(
            "TOOL_PROVENANCE_REQUIRED",
            f"{node_kind.value} must cite a tool result",
        )
    records = []
    for ref in refs:
        record = runtime.tool_results.get(ref)
        if record is None:
            raise GraphValidationError("UNKNOWN_TOOL_RESULT", ref)
        if record.action_id != action_id or record.sub_traj_id != sub_traj_id:
            raise GraphValidationError(
                "TOOL_RESULT_OWNER_MISMATCH",
                f"Tool result {ref!r} is not owned by the current action/sub-trajectory",
            )
        if record.tool_name not in {"search", "access"}:
            raise GraphValidationError(
                "INVALID_TOOL_PROVENANCE",
                f"Unsupported source tool {record.tool_name!r}",
            )
        if not record.success:
            raise GraphValidationError(
                "INVALID_TOOL_PROVENANCE",
                f"Tool result {ref!r} did not complete successfully",
            )
        records.append(record)
    if node_kind == EvidenceKind.SOURCE and not any(
        record.tool_name == "access" for record in records
    ):
        raise GraphValidationError(
            "SOURCE_ACCESS_REQUIRED", "Source must cite an access result"
        )
    return refs


def _v2_validate_source_hash(
    runtime: GraphRuntime, node: EvidenceNode, refs: tuple[str, ...]
) -> None:
    if node.kind != EvidenceKind.SOURCE:
        return
    uri = str(node.payload.get("uri", ""))
    content_hash = node.payload.get("content_hash")
    records = [runtime.tool_results[ref] for ref in refs]
    access_records = [record for record in records if record.tool_name == "access"]
    if not isinstance(content_hash, str) or not content_hash.strip():
        raise GraphValidationError(
            "SOURCE_HASH_REQUIRED", "Source payload requires content_hash"
        )
    if any(record.url and record.url != uri for record in access_records):
        raise GraphValidationError("SOURCE_URI_MISMATCH", uri)
    if not any(str(content_hash) == record.result_hash for record in access_records):
        raise GraphValidationError(
            "SOURCE_HASH_MISMATCH",
            f"Source {node.node_id!r} has an invalid content_hash",
        )


def _v2_sources_for_node(runtime: GraphRuntime, node_id: str) -> set[str]:
    """Walk the active support chain and return source IDs."""

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
        for edge in graph.incoming(current, active_only=True):
            if edge.relation in {"SUPPORTED_BY", "OBSERVED_IN", "MENTIONS"}:
                queue.append(edge.source_id)
        for edge in graph.outgoing(current, active_only=True):
            if edge.relation in {"OBSERVED_IN", "SUPPORTED_BY"}:
                queue.append(edge.target_id)
    return found


def _v2_validate_fact_provenance(runtime: GraphRuntime, fact: EvidenceNode) -> None:
    if (
        fact.kind != EvidenceKind.FACT
        or not runtime.config.require_source_for_verified_claim
    ):
        return
    explicit = fact.payload.get("source_refs", fact.payload.get("basis_refs", ()))
    explicit_refs = (
        tuple(str(ref) for ref in explicit)
        if isinstance(explicit, (list, tuple))
        else ()
    )
    source_ids = set()
    for ref in explicit_refs:
        source_id = _v2_resolve_ref(runtime, ref)
        if runtime.evidence_graph.nodes[source_id].kind != EvidenceKind.SOURCE:
            raise GraphValidationError(
                "FACT_SOURCE_REQUIRED", f"{ref!r} is not a Source"
            )
        source_ids.add(source_id)
    claim_ref = fact.payload.get("claim_ref")
    if claim_ref:
        source_ids.update(
            _v2_sources_for_node(runtime, _v2_resolve_ref(runtime, str(claim_ref)))
        )
    if len(source_ids) < runtime.config.min_independent_sources:
        raise GraphValidationError(
            "FACT_PROVENANCE_REQUIRED", "Fact requires active Source provenance"
        )


def _v2_potential_conflict_keys(runtime: GraphRuntime, claim_ids: set[str]) -> set[str]:
    graph = runtime.evidence_graph
    groups: dict[tuple[Any, Any], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for claim in graph.iter_kind(EvidenceKind.CLAIM):
        subject = claim.payload.get("subject_ref")
        predicate = claim.payload.get("predicate")
        if claim.active and subject is not None and predicate is not None:
            groups[(str(subject), str(predicate))][
                repr(claim.payload.get("object", claim.payload.get("value")))
            ].add(claim.node_id)
    result = set()
    for (subject, predicate), values_by_value in groups.items():
        group_claim_ids = set().union(*values_by_value.values())
        if len(values_by_value) > 1 and group_claim_ids.intersection(claim_ids):
            result.add(_stable_id("pending-conflict", f"{subject}|{predicate}"))
    return result


async def bootstrap_entities(
    runtime: GraphRuntime,
    proposals: list[NodeProposal] | tuple[NodeProposal, ...],
) -> CommitResult:
    """Commit the one isolated, ground-truth-free Entity bootstrap batch."""

    async with runtime.lock:
        if runtime.bootstrap_metadata.get("called"):
            raise GraphValidationError(
                "BOOTSTRAP_ALREADY_CALLED", "Entity bootstrap is one-shot"
            )
        normalized = tuple(
            item
            if isinstance(item, NodeProposal)
            else NodeProposal(
                client_ref=str(
                    item.get("client_ref", item.get("ref", f"entity_{index}"))
                ),
                kind=EvidenceKind.ENTITY,
                canonical_key=str(
                    item.get("canonical_key", item.get("canonical_name", ""))
                ),
                payload=dict(item.get("payload", item)),
            )
            for index, item in enumerate(proposals)
        )
        if any(item.normalized_kind() != EvidenceKind.ENTITY for item in normalized):
            raise GraphValidationError(
                "BOOTSTRAP_ENTITY_ONLY", "Bootstrap accepts only Entity proposals"
            )
        operations = [
            {
                "op": "add_node",
                "node": {**item.__dict__, "kind": EvidenceKind.ENTITY.value},
            }
            for item in normalized
        ]
        if not operations:
            runtime.bootstrap_metadata["called"] = True
            runtime.bootstrap_metadata["entity_count"] = 0
            return CommitResult("bootstrap:empty", runtime.version, GraphDelta())
        result = await _commit_edit_mem_locked(
            runtime,
            base_version=runtime.version,
            operations=operations,
            actor_role="main",
            event_type=GraphEventType.BOOTSTRAP_ENTITY,
            proposal_id="bootstrap:entities",
        )
        runtime.bootstrap_metadata.update(
            {"called": True, "entity_count": len(result.delta.node_ids)}
        )
        runtime.bootstrap_entities = tuple(result.delta.node_ids)
        return result


async def commit_add_mem(
    runtime: GraphRuntime,
    proposal: EvidenceProposal,
) -> CommitResult:
    """Validate and atomically commit a Subagent ``add_mem`` batch."""

    runtime.pending_memory_transaction = True
    try:
        async with runtime.lock:
            snapshot = _v2_snapshot(runtime)
            try:
                if proposal.created_by_role != "subagent":
                    raise GraphValidationError(
                        "ADD_MEM_ROLE_REQUIRED",
                        "add_mem must be submitted by a subagent",
                    )
                action = runtime.activation_dag.actions.get(proposal.action_id)
                if action is None:
                    raise GraphValidationError("UNKNOWN_ACTION", proposal.action_id)
                if action.state not in {ActionState.READY, ActionState.RUNNING}:
                    raise GraphValidationError("ACTION_NOT_RUNNING", proposal.action_id)
                if action.owner_sub_traj not in {None, proposal.created_by_sub_traj}:
                    raise GraphValidationError(
                        "ACTION_OWNER_MISMATCH", proposal.action_id
                    )
                if proposal.base_version != runtime.version:
                    raise GraphValidationError(
                        "VERSION_CONFLICT",
                        f"Proposal base version {proposal.base_version}; current graph version {runtime.version}",
                    )
                if not proposal.nodes:
                    raise GraphValidationError(
                        "EMPTY_ADD_MEM", "add_mem requires Source or Candidate nodes"
                    )
                refs: set[str] = set()
                for node_proposal in proposal.nodes:
                    kind = node_proposal.normalized_kind()
                    if kind not in _V2_SUBAGENT_NODE_KINDS:
                        raise GraphValidationError(
                            "SUBAGENT_NODE_FORBIDDEN",
                            f"Subagents cannot propose {kind.value}",
                        )
                    if node_proposal.client_ref in refs or not node_proposal.client_ref:
                        raise GraphValidationError(
                            "DUPLICATE_CLIENT_REF", node_proposal.client_ref
                        )
                    refs.add(node_proposal.client_ref)
                    node_refs = tuple(node_proposal.tool_result_refs) or tuple(
                        proposal.tool_result_refs
                    )
                    _v2_tool_refs(
                        runtime,
                        node_refs,
                        action_id=proposal.action_id,
                        sub_traj_id=proposal.created_by_sub_traj,
                        node_kind=kind,
                    )
                    candidate = _v2_node_from_proposal(
                        runtime,
                        node_proposal,
                        actor_role="subagent",
                        action_id=proposal.action_id,
                        sub_traj_id=proposal.created_by_sub_traj,
                        version=runtime.version + 1,
                    )
                    _v2_validate_source_hash(runtime, candidate, node_refs)
                    existing = runtime.evidence_graph.get_by_canonical(
                        candidate.canonical_key, kind
                    )
                    if (
                        existing is not None
                        and kind == EvidenceKind.SOURCE
                        and existing.payload != candidate.payload
                    ):
                        raise GraphValidationError("SOURCE_IMMUTABLE", existing.node_id)
                if action.state == ActionState.READY:
                    runtime.mark_action_running(
                        proposal.action_id, owner_sub_traj=proposal.created_by_sub_traj
                    )
                return _commit_add_mem_locked(runtime, proposal)
            except Exception:
                _v2_restore(runtime, snapshot)
                raise
    finally:
        runtime.pending_memory_transaction = False


def _commit_add_mem_locked(
    runtime: GraphRuntime, proposal: EvidenceProposal
) -> CommitResult:
    snapshot = _v2_snapshot(runtime)
    graph = runtime.evidence_graph
    mapping: dict[str, str] = {}
    changed_nodes: list[str] = []
    changed_edges: list[str] = []
    all_tool_result_refs = tuple(
        dict.fromkeys(
            [
                *proposal.tool_result_refs,
                *(ref for node in proposal.nodes for ref in node.tool_result_refs),
            ]
        )
    )
    try:
        for node_proposal in proposal.nodes:
            kind = node_proposal.normalized_kind()
            candidate = _v2_node_from_proposal(
                runtime,
                node_proposal,
                actor_role="subagent",
                action_id=proposal.action_id,
                sub_traj_id=proposal.created_by_sub_traj,
                version=runtime.version + 1,
            )
            refs = tuple(node_proposal.tool_result_refs) or tuple(
                proposal.tool_result_refs
            )
            candidate = EvidenceNode(**{**candidate.__dict__, "tool_result_refs": refs})
            existing = graph.get_by_canonical(candidate.canonical_key, kind)
            if existing is not None:
                mapping[node_proposal.client_ref] = existing.node_id
                continue
            canonical, inserted = graph.add_node(candidate)
            mapping[node_proposal.client_ref] = canonical.node_id
            if inserted:
                changed_nodes.append(canonical.node_id)

        for node_proposal in proposal.nodes:
            node_id = mapping[node_proposal.client_ref]
            node = graph.nodes[node_id]
            payload = _remap_value(copy.deepcopy(node_proposal.payload), mapping)
            if payload and node.kind != EvidenceKind.SOURCE:
                graph.replace_node(
                    EvidenceNode(
                        **{
                            **node.__dict__,
                            "payload": {**node.payload, **payload},
                            "updated_at_version": runtime.version + 1,
                        }
                    )
                )

        for edge_proposal in proposal.edges:
            source_id = _v2_resolve_ref(runtime, edge_proposal.source_ref, mapping)
            target_id = _v2_resolve_ref(runtime, edge_proposal.target_ref, mapping)
            _v2_check_edge(
                runtime,
                str(edge_proposal.relation).upper(),
                source_id,
                target_id,
                actor_role="subagent",
            )
            edge = EvidenceEdge(
                edge_id=_stable_id(
                    "edge",
                    f"{source_id}|{str(edge_proposal.relation).upper()}|{target_id}",
                ),
                source_id=source_id,
                relation=str(edge_proposal.relation).upper(),
                target_id=target_id,
                created_by_action=proposal.action_id,
                created_at_version=runtime.version + 1,
                created_by_role="subagent",
                proposed_by_turn=runtime.main_turn,
            )
            _, inserted = graph.add_edge(edge)
            if inserted:
                changed_edges.append(edge.edge_id)

        if (
            len(graph.nodes) > runtime.config.max_nodes
            or len(graph.edges) > runtime.config.max_edges
        ):
            raise GraphValidationError("GRAPH_LIMIT", "Evidence graph budget exceeded")
        runtime.version += 1
        graph.version = runtime.version
        runtime.action_results.setdefault(proposal.action_id, {}).update(
            {
                "proposal_id": proposal.proposal_id,
                "accepted_node_count": len(changed_nodes),
                "accepted_edge_count": len(changed_edges),
                "tool_result_refs": all_tool_result_refs,
            }
        )
        status = proposal.action_result.status.lower()
        if status in {"completed", "failed", "blocked", "invalidated"}:
            runtime.mark_action_completed(
                proposal.action_id,
                status=status,
                summary=proposal.action_result.summary,
                partition=proposal.action_result.partition_cursor,
            )
        event_ids = []
        if changed_nodes:
            event_ids.append(
                runtime._append_event(
                    GraphEventType.ADD_NODE,
                    actor_role="subagent",
                    action_id=proposal.action_id,
                    sub_traj_id=proposal.created_by_sub_traj,
                    node_ids=tuple(changed_nodes),
                    tool_result_refs=all_tool_result_refs,
                ).event_id
            )
        if changed_edges:
            event_ids.append(
                runtime._append_event(
                    GraphEventType.ADD_EDGE,
                    actor_role="subagent",
                    action_id=proposal.action_id,
                    sub_traj_id=proposal.created_by_sub_traj,
                    edge_ids=tuple(changed_edges),
                    tool_result_refs=all_tool_result_refs,
                ).event_id
            )
        return CommitResult(
            proposal_id=proposal.proposal_id,
            graph_version=runtime.version,
            delta=GraphDelta(
                node_ids=tuple(changed_nodes), edge_ids=tuple(changed_edges)
            ),
            transitions=tuple({"event_id": event_id} for event_id in event_ids),
        )
    except Exception:
        _v2_restore(runtime, snapshot)
        raise


async def commit_edit_mem(
    runtime: GraphRuntime,
    *,
    base_version: int,
    operations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    actor_role: str = "main",
    proposal_id: str = "edit:unknown",
) -> CommitResult:
    """Validate and atomically apply one Main ``edit_mem`` transaction."""

    if actor_role != "main":
        raise GraphValidationError("EDIT_MEM_ROLE_REQUIRED", "edit_mem is Main-only")
    runtime.pending_memory_transaction = True
    try:
        async with runtime.lock:
            return await _commit_edit_mem_locked(
                runtime,
                base_version=base_version,
                operations=operations,
                actor_role=actor_role,
                proposal_id=proposal_id,
            )
    finally:
        runtime.pending_memory_transaction = False


async def _commit_edit_mem_locked(
    runtime: GraphRuntime,
    *,
    base_version: int,
    operations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    actor_role: str,
    proposal_id: str,
    event_type: GraphEventType = GraphEventType.ADD_NODE,
) -> CommitResult:
    if actor_role != "main":
        raise GraphValidationError("EDIT_MEM_ROLE_REQUIRED", "edit_mem is Main-only")
    if base_version != runtime.version:
        raise GraphValidationError(
            "VERSION_CONFLICT",
            f"Edit base version {base_version}; current graph version {runtime.version}",
        )
    if not operations:
        raise GraphValidationError("EMPTY_EDIT_MEM", "edit_mem requires operations")
    snapshot = _v2_snapshot(runtime)
    graph = runtime.evidence_graph
    mapping: dict[str, str] = {}
    changed_nodes: list[str] = []
    changed_edges: list[str] = []
    fact_ids: list[str] = []
    conflict_ids: list[str] = []
    new_claim_ids: set[str] = set()
    operation_events: list[tuple[GraphEventType, dict[str, Any]]] = []
    try:
        normalized_ops = [dict(operation) for operation in operations]
        for operation in normalized_ops:
            op = str(operation.get("op", operation.get("operation", ""))).lower()
            if op in {"add_node", "add"}:
                value = dict(operation.get("node", operation))
                proposal = NodeProposal(
                    client_ref=str(value.get("client_ref", value.get("ref", ""))),
                    kind=value.get("kind", ""),
                    canonical_key=str(value.get("canonical_key", value.get("key", ""))),
                    payload=_remap_value(dict(value.get("payload", {})), mapping),
                    status=value.get("status", EvidenceStatus.PROPOSED.value),
                    confidence=value.get("confidence"),
                    tags=tuple(str(tag) for tag in value.get("tags", ())),
                )
                kind = proposal.normalized_kind()
                if kind not in _V2_MAIN_NODE_KINDS:
                    raise GraphValidationError(
                        "MAIN_NODE_FORBIDDEN", f"Main cannot propose {kind.value}"
                    )
                if not proposal.client_ref or not proposal.canonical_key:
                    raise GraphValidationError(
                        "INVALID_NODE", "client_ref and canonical_key are required"
                    )
                node = _v2_node_from_proposal(
                    runtime,
                    proposal,
                    actor_role="main",
                    action_id="main:edit_mem",
                    sub_traj_id=0,
                    version=runtime.version + 1,
                )
                existing = graph.get_by_canonical(node.canonical_key, kind)
                if existing is not None:
                    mapping[proposal.client_ref] = existing.node_id
                    if kind == EvidenceKind.SOURCE and existing.payload != node.payload:
                        raise GraphValidationError("SOURCE_IMMUTABLE", existing.node_id)
                    continue
                canonical, inserted = graph.add_node(node)
                mapping[proposal.client_ref] = canonical.node_id
                if inserted:
                    changed_nodes.append(canonical.node_id)
                    if kind == EvidenceKind.CLAIM:
                        new_claim_ids.add(canonical.node_id)
                        operation_events.append(
                            (GraphEventType.ADD_NODE, {"node_kind": kind.value})
                        )
                    elif kind == EvidenceKind.CONFLICT:
                        conflict_ids.append(canonical.node_id)
            elif op in {"add_edge", "link"}:
                source_id = _v2_resolve_ref(
                    runtime,
                    str(operation.get("source_ref", operation.get("from", ""))),
                    mapping,
                )
                target_id = _v2_resolve_ref(
                    runtime,
                    str(operation.get("target_ref", operation.get("to", ""))),
                    mapping,
                )
                relation = str(
                    operation.get("relation", operation.get("type", ""))
                ).upper()
                _v2_check_edge(
                    runtime, relation, source_id, target_id, actor_role="main"
                )
                edge = EvidenceEdge(
                    edge_id=_stable_id("edge", f"{source_id}|{relation}|{target_id}"),
                    source_id=source_id,
                    relation=relation,
                    target_id=target_id,
                    created_by_action="main:edit_mem",
                    created_at_version=runtime.version + 1,
                    created_by_role="main",
                    proposed_by_turn=runtime.main_turn,
                )
                _, inserted = graph.add_edge(edge)
                if inserted:
                    changed_edges.append(edge.edge_id)
            elif op in {"update_metadata", "update_node"}:
                node_id = _v2_resolve_ref(
                    runtime,
                    str(operation.get("ref", operation.get("node_ref", ""))),
                    mapping,
                )
                node = graph.nodes[node_id]
                if node.kind == EvidenceKind.SOURCE:
                    raise GraphValidationError("SOURCE_IMMUTABLE", node_id)
                patch = dict(operation.get("patch", operation.get("metadata", {})))
                requested_status = patch.get("status", operation.get("status"))
                if requested_status is not None:
                    try:
                        status = EvidenceStatus(str(requested_status).lower())
                    except ValueError as exc:
                        raise GraphValidationError(
                            "INVALID_NODE_STATUS", str(requested_status)
                        ) from exc
                    if (
                        node.kind == EvidenceKind.SOURCE
                        and status != EvidenceStatus.RETIRED
                    ):
                        raise GraphValidationError(
                            "SOURCE_STATUS_FORBIDDEN",
                            "Source raw lifecycle can only be retired",
                        )
                else:
                    status = node.status
                updated = EvidenceNode(
                    **{
                        **node.__dict__,
                        "payload": {**node.payload, **dict(patch.get("payload", {}))},
                        "status": status,
                        "active": status
                        not in {
                            EvidenceStatus.REJECTED,
                            EvidenceStatus.INVALIDATED,
                            EvidenceStatus.RETIRED,
                            EvidenceStatus.MERGED,
                        },
                        "updated_at_version": runtime.version + 1,
                    }
                )
                graph.replace_node(updated)
                changed_nodes.append(node_id)
                if node.kind == EvidenceKind.CLAIM:
                    if updated.active and updated.status in {
                        EvidenceStatus.PENDING,
                        EvidenceStatus.VERIFYING,
                    }:
                        runtime.pending_claim_ids.add(node_id)
                    else:
                        runtime.pending_claim_ids.discard(node_id)
                if node.kind == EvidenceKind.CONFLICT:
                    if updated.active and updated.status in {
                        EvidenceStatus.OPEN,
                        EvidenceStatus.INVESTIGATING,
                    }:
                        runtime.pending_conflict_ids.add(node_id)
                    else:
                        runtime.pending_conflict_ids.discard(node_id)
            elif op in {"retire_node", "retire"}:
                node_id = _v2_resolve_ref(
                    runtime,
                    str(operation.get("ref", operation.get("node_ref", ""))),
                    mapping,
                )
                graph.retire_node(node_id, version=runtime.version + 1)
                changed_nodes.append(node_id)
                runtime.pending_claim_ids.discard(node_id)
                runtime.pending_conflict_ids.discard(node_id)
                operation_events.append(
                    (
                        GraphEventType.RETIRE_NODE,
                        {
                            "node_id": node_id,
                            "reason": operation.get("reason", ""),
                        },
                    )
                )
            elif op == "retire_edge":
                edge_ref = str(operation.get("edge_id", ""))
                if edge_ref not in graph.edges:
                    source_id = _v2_resolve_ref(
                        runtime, str(operation.get("source_ref", "")), mapping
                    )
                    target_id = _v2_resolve_ref(
                        runtime, str(operation.get("target_ref", "")), mapping
                    )
                    relation = str(operation.get("relation", "")).upper()
                    edge_ref = _stable_id("edge", f"{source_id}|{relation}|{target_id}")
                edge = graph.edges.get(edge_ref)
                if edge is None:
                    raise GraphValidationError("UNKNOWN_EVIDENCE_EDGE", edge_ref)
                graph.edges[edge_ref] = EvidenceEdge(
                    **{**edge.__dict__, "active": False}
                )
                changed_edges.append(edge_ref)
                operation_events.append(
                    (
                        GraphEventType.RETIRE_EDGE,
                        {"edge_id": edge_ref},
                    )
                )
            elif op == "merge_entity":
                source_id = _v2_resolve_ref(
                    runtime, str(operation.get("source_ref", "")), mapping
                )
                target_id = _v2_resolve_ref(
                    runtime, str(operation.get("target_ref", "")), mapping
                )
                if (
                    graph.nodes[source_id].kind != EvidenceKind.ENTITY
                    or graph.nodes[target_id].kind != EvidenceKind.ENTITY
                ):
                    raise GraphValidationError(
                        "MERGE_ENTITY_ONLY", "merge_entity requires two Entity nodes"
                    )
                merged = graph.nodes[source_id].with_status(
                    EvidenceStatus.MERGED, version=runtime.version + 1
                )
                graph.replace_node(merged)
                edge = EvidenceEdge(
                    edge_id=_stable_id("edge", f"{source_id}|SAME_AS|{target_id}"),
                    source_id=source_id,
                    relation="SAME_AS",
                    target_id=target_id,
                    created_by_action="main:edit_mem",
                    created_at_version=runtime.version + 1,
                    created_by_role="main",
                    proposed_by_turn=runtime.main_turn,
                )
                _, inserted = graph.add_edge(edge)
                if inserted:
                    changed_edges.append(edge.edge_id)
                changed_nodes.append(source_id)
                operation_events.append(
                    (
                        GraphEventType.MERGE_ENTITY,
                        {"source_id": source_id, "target_id": target_id},
                    )
                )
            elif op == "promote_claim":
                claim_id = _v2_resolve_ref(
                    runtime, str(operation.get("claim_ref", "")), mapping
                )
                claim = graph.nodes[claim_id]
                if claim.kind != EvidenceKind.CLAIM:
                    raise GraphValidationError("CLAIM_REQUIRED", claim_id)
                if (
                    runtime.config.require_claim_next_turn_verification
                    and claim.proposed_by_turn >= runtime.main_turn
                ):
                    raise GraphValidationError("SAME_TURN_PROMOTION", claim_id)
                fact_value = dict(operation.get("fact", {}))
                fact_value.setdefault("claim_ref", claim_id)
                fact_value.setdefault("source_refs", operation.get("source_refs", ()))
                fact_proposal = NodeProposal(
                    client_ref=str(operation.get("fact_ref", f"fact_for_{claim_id}")),
                    kind=EvidenceKind.FACT,
                    canonical_key=str(
                        operation.get(
                            "canonical_key", f"{claim.canonical_key}:verified"
                        )
                    ),
                    payload=fact_value,
                    status=EvidenceStatus.VERIFIED,
                    confidence=operation.get("confidence", claim.confidence),
                )
                fact = _v2_node_from_proposal(
                    runtime,
                    fact_proposal,
                    actor_role="main",
                    action_id="main:edit_mem",
                    sub_traj_id=0,
                    version=runtime.version + 1,
                )
                _v2_validate_fact_provenance(runtime, fact)
                existing = graph.get_by_canonical(fact.canonical_key, EvidenceKind.FACT)
                if existing is None:
                    graph.add_node(fact)
                    fact_ids.append(fact.node_id)
                    changed_nodes.append(fact.node_id)
                    existing = fact
                mapping[fact_proposal.client_ref] = existing.node_id
                edge = EvidenceEdge(
                    edge_id=_stable_id(
                        "edge", f"{claim_id}|VERIFIED_AS|{existing.node_id}"
                    ),
                    source_id=claim_id,
                    relation="VERIFIED_AS",
                    target_id=existing.node_id,
                    created_by_action="main:edit_mem",
                    created_at_version=runtime.version + 1,
                    created_by_role="main",
                    proposed_by_turn=runtime.main_turn,
                )
                _, inserted = graph.add_edge(edge)
                if inserted:
                    changed_edges.append(edge.edge_id)
                # Phase 2/4 render and terminal checks use a direct Fact ->
                # Source projection.  Keep the Claim -> Fact provenance edge
                # above as well; both are immutable evidence lineage.
                source_refs = fact.payload.get("source_refs", ())
                if isinstance(source_refs, (list, tuple)):
                    for source_ref in source_refs:
                        source_id = _v2_resolve_ref(runtime, str(source_ref))
                        if graph.nodes[source_id].kind != EvidenceKind.SOURCE:
                            raise GraphValidationError(
                                "FACT_SOURCE_REQUIRED", str(source_ref)
                            )
                        source_edge = EvidenceEdge(
                            edge_id=_stable_id(
                                "edge", f"{existing.node_id}|SUPPORTED_BY|{source_id}"
                            ),
                            source_id=existing.node_id,
                            relation="SUPPORTED_BY",
                            target_id=source_id,
                            created_by_action="main:edit_mem",
                            created_at_version=runtime.version + 1,
                            created_by_role="main",
                            proposed_by_turn=runtime.main_turn,
                        )
                        _, source_inserted = graph.add_edge(source_edge)
                        if source_inserted:
                            changed_edges.append(source_edge.edge_id)
                graph.replace_node(
                    claim.with_status(
                        EvidenceStatus.PROMOTED, version=runtime.version + 1
                    )
                )
                changed_nodes.append(claim_id)
                runtime.pending_claim_ids.discard(claim_id)
                operation_events.append(
                    (
                        GraphEventType.PROMOTE_CLAIM,
                        {"claim_id": claim_id, "fact_id": existing.node_id},
                    )
                )
            elif op == "resolve_conflict":
                conflict_id = _v2_resolve_ref(
                    runtime, str(operation.get("conflict_ref", "")), mapping
                )
                conflict = graph.nodes[conflict_id]
                if conflict.kind != EvidenceKind.CONFLICT:
                    raise GraphValidationError("CONFLICT_REQUIRED", conflict_id)
                winner = operation.get("winner_ref")
                competing = conflict.payload.get("competing_refs", [])
                competing_ids = {
                    _v2_resolve_ref(runtime, str(ref), mapping) for ref in competing
                }
                if not winner:
                    raise GraphValidationError("CONFLICT_WINNER_REQUIRED", conflict_id)
                winner_id = _v2_resolve_ref(runtime, str(winner), mapping)
                if competing_ids and winner_id not in competing_ids:
                    raise GraphValidationError("CONFLICT_WINNER_INVALID", winner_id)
                retire_ids = {
                    _v2_resolve_ref(runtime, str(ref), mapping)
                    for ref in operation.get("retire_refs", [])
                }
                required_retire_ids = competing_ids - {winner_id}
                if required_retire_ids != retire_ids:
                    raise GraphValidationError(
                        "CONFLICT_RESOLUTION_INCOMPLETE",
                        f"Expected losers {sorted(required_retire_ids)}, got {sorted(retire_ids)}",
                    )
                for ref_id in sorted(retire_ids):
                    graph.retire_node(ref_id, version=runtime.version + 1)
                    changed_nodes.append(ref_id)
                graph.replace_node(
                    EvidenceNode(
                        **{
                            **conflict.with_status(
                                EvidenceStatus.RESOLVED, version=runtime.version + 1
                            ).__dict__,
                            "active": False,
                        }
                    )
                )
                changed_nodes.append(conflict_id)
                runtime.pending_conflict_ids.discard(conflict_id)
                operation_events.append(
                    (
                        GraphEventType.RESOLVE_CONFLICT,
                        {"conflict_id": conflict_id, "winner_id": winner_id},
                    )
                )
            else:
                raise GraphValidationError("UNKNOWN_EDIT_OPERATION", op)

        if (
            any(
                graph.nodes[node_id].kind == EvidenceKind.FACT
                for node_id in changed_nodes
                if node_id in graph.nodes
            )
            and new_claim_ids
        ):
            raise GraphValidationError(
                "SAME_TURN_PROMOTION",
                "A transaction cannot add a Claim and Fact together",
            )
        for node_id in changed_nodes:
            node = graph.nodes.get(node_id)
            if node is not None and node.kind == EvidenceKind.FACT:
                _v2_validate_fact_provenance(runtime, node)
        if (
            len(graph.nodes) > runtime.config.max_nodes
            or len(graph.edges) > runtime.config.max_edges
        ):
            raise GraphValidationError("GRAPH_LIMIT", "Evidence graph budget exceeded")

        runtime.version += 1
        graph.version = runtime.version
        runtime.pending_claim_ids.update(new_claim_ids)
        runtime.pending_claim_ids = {
            node_id
            for node_id in runtime.pending_claim_ids
            if node_id in graph.nodes and graph.nodes[node_id].active
        }
        potential_conflicts = _v2_potential_conflict_keys(runtime, new_claim_ids)
        runtime.pending_conflict_ids.update(conflict_ids)
        runtime.pending_conflict_ids.update(potential_conflicts)
        if len(runtime.pending_claim_ids) > runtime.config.max_pending_claims:
            raise GraphValidationError(
                "PENDING_CLAIM_LIMIT", "Pending Claim queue budget exceeded"
            )
        if len(runtime.pending_conflict_ids) > runtime.config.max_pending_conflicts:
            raise GraphValidationError(
                "PENDING_CONFLICT_LIMIT", "Pending Conflict queue budget exceeded"
            )
        event_ids = []
        if changed_nodes:
            event_ids.append(
                runtime._append_event(
                    event_type
                    if event_type != GraphEventType.ADD_NODE
                    else GraphEventType.ADD_NODE,
                    actor_role="main",
                    node_ids=tuple(dict.fromkeys(changed_nodes)),
                    payload={"operation_count": len(normalized_ops)},
                ).event_id
            )
        if changed_edges:
            event_ids.append(
                runtime._append_event(
                    GraphEventType.ADD_EDGE,
                    actor_role="main",
                    edge_ids=tuple(dict.fromkeys(changed_edges)),
                ).event_id
            )
        for kind, payload in operation_events:
            node_event_ids = tuple(
                str(payload[key])
                for key in (
                    "node_id",
                    "source_id",
                    "target_id",
                    "claim_id",
                    "fact_id",
                    "conflict_id",
                    "winner_id",
                )
                if payload.get(key)
            )
            edge_event_ids = (
                (str(payload["edge_id"]),) if payload.get("edge_id") else ()
            )
            event_ids.append(
                runtime._append_event(
                    kind,
                    actor_role="main",
                    node_ids=node_event_ids,
                    edge_ids=edge_event_ids,
                    payload=payload,
                ).event_id
            )
        for conflict_key in sorted(potential_conflicts):
            event_ids.append(
                runtime._append_event(
                    GraphEventType.POTENTIAL_CONFLICT,
                    actor_role="system",
                    payload={"pending_conflict_id": conflict_key},
                ).event_id
            )
        return CommitResult(
            proposal_id=proposal_id,
            graph_version=runtime.version,
            delta=GraphDelta(
                node_ids=tuple(dict.fromkeys(changed_nodes)),
                edge_ids=tuple(dict.fromkeys(changed_edges)),
                fact_ids=tuple(dict.fromkeys(fact_ids)),
                conflict_ids=tuple(dict.fromkeys(conflict_ids)),
            ),
            transitions=tuple({"event_id": event_id} for event_id in event_ids),
        )
    except Exception:
        _v2_restore(runtime, snapshot)
        raise


async def commit_evidence(
    runtime: GraphRuntime,
    proposal: EvidenceProposal,
    *,
    validator: GraphValidator | None = None,
) -> CommitResult:
    """Validate and atomically commit an evidence proposal."""

    validator = validator or GraphValidator()
    async with runtime.lock:
        return validator._commit_locked(runtime, proposal)


async def compile_task_plan(
    runtime: GraphRuntime,
    proposal: TaskPlanProposal,
    *,
    validator: GraphValidator | None = None,
) -> TaskContract:
    """Validate and atomically install a Task Contract/Activation plan."""

    validator = validator or GraphValidator()
    async with runtime.lock:
        normalized = validator.validate_task_plan(runtime, proposal)
        if runtime.version >= runtime.config.max_graph_versions:
            raise GraphValidationError(
                "GRAPH_VERSION_LIMIT", "graph version budget exceeded"
            )
        rollback = {
            "evidence_graph": copy.deepcopy(runtime.evidence_graph),
            "activation_dag": copy.deepcopy(runtime.activation_dag),
            "contract": runtime.contract,
            "version": runtime.version,
            "plan_version": runtime.plan_version,
            "event_queues": copy.deepcopy(runtime.event_queues),
            "emitted_event_keys": set(runtime.emitted_event_keys),
            "events_by_key": copy.deepcopy(runtime.events_by_key),
            "delivered_event_keys": set(runtime.delivered_event_keys),
            "action_acl": copy.deepcopy(runtime.action_acl),
            "action_acl_expiry": dict(runtime.action_acl_expiry),
            "action_results": copy.deepcopy(runtime.action_results),
            "covered_partitions": set(runtime.covered_partitions),
            "remaining_budget": runtime.remaining_budget,
            "last_error": runtime.last_error,
        }
        try:
            runtime.version += 1
            runtime.evidence_graph.version = runtime.version
            # Anchor entities are semantic proposals from the main agent, but
            # the system still canonicalizes and owns their IDs. They must be
            # present before activation payloads are materialized.
            for anchor in normalized.anchor_entities:
                node = EvidenceNode(
                    node_id=_stable_id("evidence:entity", anchor.canonical_key),
                    kind=EvidenceKind.ENTITY,
                    canonical_key=anchor.canonical_key,
                    payload=copy.deepcopy(anchor.payload),
                    status=EvidenceStatus.PROPOSED,
                    created_by_action="action:plan_task",
                    created_at_version=runtime.version,
                    tags=tuple(anchor.tags),
                )
                if (
                    runtime.evidence_graph.get_by_canonical(
                        anchor.canonical_key, EvidenceKind.ENTITY
                    )
                    is None
                ):
                    runtime.evidence_graph.add_node(node)
            runtime.install_plan(
                normalized.contract
                if isinstance(normalized.contract, TaskContract)
                else TaskContract.from_dict(
                    normalized.contract, question=runtime.question
                ),
                actions=normalized.actions,
                gates=normalized.gates,
                payloads=normalized.payloads,
                joins=normalized.joins,
                audits=normalized.audits,
                renders=normalized.renders,
                edges=normalized.edges,
            )
            runtime.evaluate_activation()
            return runtime.contract
        except Exception:
            for field_name, value in rollback.items():
                setattr(runtime, field_name, value)
            raise
