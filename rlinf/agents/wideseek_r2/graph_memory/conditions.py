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

"""Whitelist-based Gate DSL for the Activation DAG."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.schema import EvidenceKind, EvidenceStatus


class ConditionError(ValueError):
    """Raised when a Gate condition is malformed or uses an unknown operator."""


ALLOWED_OPERATORS = frozenset(
    {
        "fact_exists",
        "claim_status",
        "entity_count",
        "all_fields_present",
        "no_open_conflict",
        "actions_completed",
        "partition_coverage",
        "source_count_at_least",
        "budget_remaining",
        "all",
        "any",
        "not",
        "true",
        "false",
    }
)


def _op(condition: Mapping[str, Any]) -> str:
    value = condition.get("op", condition.get("type"))
    if value is None:
        raise ConditionError("Gate condition requires an 'op'")
    value = str(value).lower()
    if value not in ALLOWED_OPERATORS:
        raise ConditionError(
            f"Unsupported Gate operator {value!r}; allowed operators are "
            f"{sorted(ALLOWED_OPERATORS)}"
        )
    return value


def validate_condition(condition: Any) -> dict[str, Any]:
    """Validate and normalize a condition without evaluating it.

    The compiler intentionally accepts only JSON-like dictionaries and the
    operators in :data:`ALLOWED_OPERATORS`; arbitrary Python expressions are
    never evaluated.
    """

    if isinstance(condition, bool):
        return {"op": "true" if condition else "false"}
    if not isinstance(condition, Mapping):
        raise ConditionError("Gate condition must be a JSON object")
    operator = _op(condition)
    normalized = dict(condition)
    normalized["op"] = operator
    if operator in {"all", "any"}:
        children = condition.get("conditions", condition.get("args", []))
        if not isinstance(children, (list, tuple)):
            raise ConditionError(f"{operator} requires a list of conditions")
        normalized["conditions"] = [validate_condition(item) for item in children]
    elif operator == "not":
        child = condition.get("condition", condition.get("arg"))
        normalized["condition"] = validate_condition(child)
    elif operator == "fact_exists":
        if not condition.get("ref") and not condition.get("canonical_key"):
            raise ConditionError("fact_exists requires ref or canonical_key")
    elif operator == "claim_status":
        if not condition.get("ref") and not condition.get("canonical_key"):
            raise ConditionError("claim_status requires ref or canonical_key")
        if not condition.get("status"):
            raise ConditionError("claim_status requires status")
    elif operator == "entity_count":
        if not any(key in condition for key in ("at_least", "at_most", "equals")):
            raise ConditionError("entity_count requires at_least, at_most, or equals")
    elif operator == "all_fields_present":
        if not condition.get("entity_ref"):
            raise ConditionError("all_fields_present requires entity_ref")
        fields = condition.get("fields", [])
        if not isinstance(fields, (list, tuple)):
            raise ConditionError("all_fields_present.fields must be a list")
    elif operator == "actions_completed":
        actions = condition.get("action_ids", condition.get("actions", []))
        if not isinstance(actions, (list, tuple)):
            raise ConditionError("actions_completed requires action_ids list")
    elif operator == "partition_coverage":
        partitions = condition.get("partitions", [])
        if not isinstance(partitions, (list, tuple)):
            raise ConditionError("partition_coverage.partitions must be a list")
    elif operator == "source_count_at_least":
        if not condition.get("claim_ref"):
            raise ConditionError("source_count_at_least requires claim_ref")
        if int(condition.get("count", 1)) < 0:
            raise ConditionError("source_count_at_least.count must be non-negative")
    elif operator == "budget_remaining":
        if int(condition.get("at_least", 1)) < 0:
            raise ConditionError("budget_remaining.at_least must be non-negative")
    return normalized


def _node(
    runtime: Any, ref: str | None = None, *, canonical_key: str | None = None
) -> Any:
    graph = runtime.evidence_graph if hasattr(runtime, "evidence_graph") else runtime
    if ref:
        return graph.nodes.get(ref) or graph.get_by_canonical(ref)
    if canonical_key:
        node_id = graph.canonical_index.get(canonical_key)
        return graph.nodes.get(node_id) if node_id else None
    return None


def _incoming(runtime: Any, target_id: str, relation: str | None = None) -> list[Any]:
    graph = runtime.evidence_graph if hasattr(runtime, "evidence_graph") else runtime
    return [
        edge
        for edge in graph.edges.values()
        if edge.target_id == target_id
        and (relation is None or edge.relation == relation)
    ]


def evaluate_condition(condition: Any, runtime: Any) -> bool:
    """Evaluate a validated Gate condition against the current runtime."""

    condition = validate_condition(condition)
    operator = condition["op"]
    graph = runtime.evidence_graph if hasattr(runtime, "evidence_graph") else runtime

    if operator == "true":
        return True
    if operator == "false":
        return False
    if operator == "all":
        return all(
            evaluate_condition(item, runtime) for item in condition["conditions"]
        )
    if operator == "any":
        return any(
            evaluate_condition(item, runtime) for item in condition["conditions"]
        )
    if operator == "not":
        return not evaluate_condition(condition["condition"], runtime)
    if operator == "fact_exists":
        node = _node(
            runtime,
            condition.get("ref"),
            canonical_key=condition.get("canonical_key"),
        )
        return bool(
            node
            and node.kind == EvidenceKind.FACT
            and node.status == EvidenceStatus.VERIFIED
        )
    if operator == "claim_status":
        node = _node(
            runtime,
            condition.get("ref"),
            canonical_key=condition.get("canonical_key"),
        )
        if node is None:
            return False
        expected = str(condition["status"]).lower()
        return (
            str(
                node.status.value if hasattr(node.status, "value") else node.status
            ).lower()
            == expected
        )
    if operator == "entity_count":
        count = sum(
            1 for node in graph.nodes.values() if node.kind == EvidenceKind.ENTITY
        )
        if "equals" in condition:
            return count == int(condition["equals"])
        if "at_most" in condition and count > int(condition["at_most"]):
            return False
        return count >= int(condition.get("at_least", 0))
    if operator == "all_fields_present":
        entity = _node(runtime, condition.get("entity_ref"))
        if entity is None:
            return False
        fields = condition.get("fields", [])
        return all(
            any(
                node.kind == EvidenceKind.FACT
                and node.status == EvidenceStatus.VERIFIED
                and node.payload.get("subject_ref") == entity.node_id
                and node.payload.get("predicate") == field
                for node in graph.nodes.values()
            )
            for field in fields
        )
    if operator == "no_open_conflict":
        refs = condition.get("refs")
        for node in graph.nodes.values():
            if node.kind != EvidenceKind.CONFLICT:
                continue
            if node.status not in {EvidenceStatus.OPEN, EvidenceStatus.CONFLICTED}:
                continue
            if (
                refs is None
                or not refs
                or any(ref in node.payload.get("competing_refs", []) for ref in refs)
            ):
                return False
        return True
    if operator == "actions_completed":
        actions = condition.get("action_ids", condition.get("actions", []))
        return all(
            runtime.activation_dag.actions.get(action_id)
            and runtime.activation_dag.actions[action_id].state
            in {runtime.action_state_completed, runtime.action_state_consumed}
            for action_id in actions
        )
    if operator == "partition_coverage":
        required = {str(item) for item in condition.get("partitions", [])}
        covered = set(getattr(runtime, "covered_partitions", set()))
        return required.issubset(covered)
    if operator == "source_count_at_least":
        claim = _node(runtime, condition.get("claim_ref"))
        if claim is None:
            return False
        source_ids = {
            edge.source_id
            for edge in _incoming(runtime, claim.node_id, "SUPPORTS")
            if graph.nodes.get(edge.source_id)
            and graph.nodes[edge.source_id].kind == EvidenceKind.SOURCE
        }
        return len(source_ids) >= int(condition.get("count", 1))
    if operator == "budget_remaining":
        return int(getattr(runtime, "remaining_budget", 0)) >= int(
            condition.get("at_least", 1)
        )
    raise ConditionError(f"Unhandled Gate operator: {operator}")


def compile_condition(condition: Any) -> dict[str, Any]:
    """Public alias used by the plan validator."""

    return validate_condition(condition)
