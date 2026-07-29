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

"""Item-task contract and terminal Fact helpers for ``mas_graph`` v2.

Phase 5 item tasks have one additional semantic invariant beyond the generic
graph invariants: the final answer must be backed by exactly one explicitly
marked, source-backed verified Fact.  Keeping the selector here gives Audit,
Render, and tests one consistent definition of the item terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.schema import (
    EvidenceKind,
    EvidenceNode,
    EvidenceStatus,
)

ITEM_TERMINAL_TAG = "terminal"


def normalize_item_format_requirements(value: Any = None) -> dict[str, Any]:
    """Return bounded, serializable defaults for an item output contract."""

    requirements = {
        "columns": ["Item"],
        "value_field": "value",
        "terminal_tag": ITEM_TERMINAL_TAG,
        "exact_rows": 1,
        "markdown_table": True,
    }
    if isinstance(value, dict):
        requirements.update(value)
    columns = requirements.get("columns", ["Item"])
    if isinstance(columns, str):
        columns = [columns]
    requirements["columns"] = [str(column) for column in columns] or ["Item"]
    requirements["value_field"] = str(requirements.get("value_field", "value"))
    requirements["terminal_tag"] = str(
        requirements.get("terminal_tag", ITEM_TERMINAL_TAG)
    )
    requirements["exact_rows"] = max(1, int(requirements.get("exact_rows", 1)))
    requirements["markdown_table"] = bool(requirements.get("markdown_table", True))
    return requirements


def _active_verified_facts(runtime: Any) -> list[EvidenceNode]:
    """Return active verified Facts in stable order."""

    return sorted(
        (
            node
            for node in runtime.evidence_graph.iter_kind(EvidenceKind.FACT)
            if node.active and node.status == EvidenceStatus.VERIFIED
        ),
        key=lambda node: node.node_id,
    )


def item_terminal_facts(runtime: Any) -> list[EvidenceNode]:
    """Select the explicitly marked terminal Facts for an item task.

    A terminal marker is intentionally required.  Treating an arbitrary
    verified intermediate Fact as the answer would allow a normal response to
    pass Audit before the final dependency hop has been resolved.
    """

    facts = _active_verified_facts(runtime)
    requirements = normalize_item_format_requirements(runtime.format_requirements)
    contract = getattr(runtime, "contract", None)
    terminal_ref = requirements.get("terminal_fact_ref") or getattr(
        contract, "terminal_fact_ref", None
    )
    if terminal_ref:
        ref = str(terminal_ref)
        node = runtime.evidence_graph.nodes.get(ref)
        if node is None:
            node = runtime.evidence_graph.get_by_canonical(ref, EvidenceKind.FACT)
        return [
            fact for fact in facts if node is not None and fact.node_id == node.node_id
        ]

    terminal_predicate = requirements.get("terminal_predicate") or getattr(
        contract, "terminal_predicate", None
    )
    if terminal_predicate:
        facts = [
            fact
            for fact in facts
            if str(fact.payload.get("predicate", "")) == str(terminal_predicate)
        ]
    if not runtime.config.item_require_terminal_fact:
        return facts
    terminal_tag = str(requirements.get("terminal_tag", ITEM_TERMINAL_TAG))
    return [
        fact
        for fact in facts
        if terminal_tag in fact.tags or bool(fact.payload.get("terminal"))
    ]


def item_value(runtime: Any, fact: EvidenceNode) -> str:
    """Extract and normalize the value rendered for one item Fact."""

    requirements = normalize_item_format_requirements(runtime.format_requirements)
    contract = getattr(runtime, "contract", None)
    field = str(requirements["value_field"])
    if field == "value" and contract is not None:
        field = str(getattr(contract, "final_value_field", field))
    value = fact.payload.get(field)
    if value in (None, "") and field != "value":
        value = fact.payload.get("value", fact.payload.get("object"))
    if value in (None, ""):
        value = fact.payload.get("object")
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class ItemCompletion:
    """Mechanical item completion result used by Audit and tests."""

    passed: bool
    terminal_refs: tuple[str, ...] = ()
    values: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def check_item_completion(runtime: Any) -> ItemCompletion:
    """Require exactly one non-empty terminal Fact for an item task."""

    facts = item_terminal_facts(runtime)
    refs = tuple(fact.node_id for fact in facts)
    values = tuple(item_value(runtime, fact) for fact in facts)
    missing: list[str] = []
    if not facts:
        missing.append("item_terminal_fact")
    if len(facts) > 1:
        missing.append("item_unique_terminal_fact")
    if facts and any(not value for value in values):
        missing.append("item_terminal_value")
    return ItemCompletion(
        passed=not missing,
        terminal_refs=refs,
        values=values,
        missing=tuple(missing),
        metadata={
            "required_rows": 1,
            "terminal_tag": normalize_item_format_requirements(
                runtime.format_requirements
            )["terminal_tag"],
        },
    )


__all__ = [
    "ITEM_TERMINAL_TAG",
    "ItemCompletion",
    "check_item_completion",
    "item_terminal_facts",
    "item_value",
    "normalize_item_format_requirements",
]
