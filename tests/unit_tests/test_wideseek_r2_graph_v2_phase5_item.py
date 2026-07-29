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

"""Phase 5 item contract tests for the existing ``mas_graph`` workflow."""

from rlinf.agents.wideseek_r2.graph_memory.audit import (
    record_audit_outcome,
    start_audit,
)
from rlinf.agents.wideseek_r2.graph_memory.item import check_item_completion
from rlinf.agents.wideseek_r2.graph_memory.prompts import GRAPH_ITEM_PLANNER_GUIDANCE
from rlinf.agents.wideseek_r2.graph_memory.renderer import (
    start_render,
    validate_render_answer,
)
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    EvidenceEdge,
    EvidenceKind,
    EvidenceNode,
    EvidenceStatus,
)
from rlinf.agents.wideseek_r2.graph_memory.state import GraphRuntime


def _runtime(**config) -> GraphRuntime:
    return GraphRuntime.bootstrap(
        question="Who leads Acme?",
        answer_type="item",
        config={"enabled": True, "schema_version": "v2", **config},
    )


def _add_grounded_fact(
    runtime: GraphRuntime,
    *,
    suffix: str,
    value: str,
    terminal: bool,
) -> str:
    """Add one source-backed Claim -> Fact chain to a test runtime."""

    graph = runtime.evidence_graph
    entity = EvidenceNode(
        node_id=f"entity:{suffix}",
        kind=EvidenceKind.ENTITY,
        canonical_key=f"entity:{suffix}",
        payload={"canonical_name": suffix},
        status=EvidenceStatus.ACTIVE,
    )
    source = EvidenceNode(
        node_id=f"source:{suffix}",
        kind=EvidenceKind.SOURCE,
        canonical_key=f"source:{suffix}",
        payload={
            "uri": f"https://example.test/{suffix}",
            "locator": "p1",
            "excerpt": f"{suffix}: {value}",
        },
        status=EvidenceStatus.ACTIVE,
    )
    claim = EvidenceNode(
        node_id=f"claim:{suffix}",
        kind=EvidenceKind.CLAIM,
        canonical_key=f"claim:{suffix}",
        payload={"subject_ref": entity.node_id, "object": value},
        status=EvidenceStatus.PROMOTED,
    )
    fact = EvidenceNode(
        node_id=f"fact:{suffix}",
        kind=EvidenceKind.FACT,
        canonical_key=f"fact:{suffix}",
        payload={
            "subject_ref": entity.node_id,
            "value": value,
            "claim_ref": claim.node_id,
            "source_refs": [source.node_id],
            "terminal": terminal,
        },
        status=EvidenceStatus.VERIFIED,
        tags=("terminal",) if terminal else (),
    )
    for node in (entity, source, claim, fact):
        graph.add_node(node)
    graph.add_edge(
        EvidenceEdge(
            f"edge:verified:{suffix}",
            claim.node_id,
            "VERIFIED_AS",
            fact.node_id,
        )
    )
    graph.add_edge(
        EvidenceEdge(
            f"edge:source:{suffix}",
            fact.node_id,
            "SUPPORTED_BY",
            source.node_id,
        )
    )
    return fact.node_id


def test_item_bootstrap_compiles_one_row_terminal_contract():
    runtime = _runtime()

    assert runtime.format_requirements == {
        "columns": ["Item"],
        "value_field": "value",
        "terminal_tag": "terminal",
        "exact_rows": 1,
        "markdown_table": True,
    }
    completion = check_item_completion(runtime)
    assert not completion.passed
    assert completion.missing == ("item_terminal_fact",)


def test_item_audit_requires_one_non_empty_terminal_fact():
    runtime = _runtime()
    _add_grounded_fact(runtime, suffix="ada", value="Ada", terminal=True)

    start_audit(runtime)
    report = record_audit_outcome(runtime, model_pass=True, response_text="AUDIT_PASS")
    assert report.passed
    assert report.invariants["item_unique_terminal_fact"]

    _add_grounded_fact(runtime, suffix="bob", value="Bob", terminal=True)
    start_audit(runtime)
    report = record_audit_outcome(runtime, model_pass=True, response_text="AUDIT_PASS")
    assert not report.passed
    assert "item_unique_terminal_fact" in report.missing


def test_item_render_payload_scopes_rows_to_the_terminal_fact():
    runtime = _runtime()
    _add_grounded_fact(runtime, suffix="intermediate", value="Ada", terminal=False)
    terminal_ref = _add_grounded_fact(
        runtime, suffix="terminal", value="Grace", terminal=True
    )

    payload = start_render(runtime)
    assert [row["fact_ref"] for row in payload["rows"]] == [terminal_ref]
    assert [fact["node_id"] for fact in payload["facts"]] == [terminal_ref]

    valid = validate_render_answer(
        "```markdown\n| Item |\n| --- |\n| Grace |\n```", runtime
    )
    assert valid.valid
    invalid = validate_render_answer(
        "```markdown\n| Item |\n| --- |\n| Grace |\n| Ada |\n```", runtime
    )
    assert not invalid.valid
    assert invalid.code == "ITEM_CARDINALITY"


def test_item_hops_are_created_dynamically_and_focus_the_next_input():
    runtime = _runtime()
    entity = EvidenceNode(
        node_id="entity:target",
        kind=EvidenceKind.ENTITY,
        canonical_key="entity:target",
        payload={"canonical_name": "Target"},
        status=EvidenceStatus.ACTIVE,
    )
    runtime.evidence_graph.add_node(entity)

    assert runtime.activation_dag.actions == {}
    first = runtime.create_action(
        "Find the intermediate entity", focus_refs=(entity.node_id,)
    )
    assert first.predecessor_ids == ()
    first_payload = runtime.activation_dag.payloads[first.payload_ids[0]]
    assert first_payload.focus_refs == (entity.node_id,)
    assert first_payload.body["nodes"][0]["node_id"] == entity.node_id

    _add_grounded_fact(runtime, suffix="intermediate", value="Ada", terminal=False)
    runtime.begin_turn(turn=1, role="planner")
    second = runtime.create_action(
        "Resolve the terminal item", focus_refs=("fact:intermediate",)
    )
    assert second.predecessor_ids == ()
    second_payload = runtime.activation_dag.payloads[second.payload_ids[0]]
    assert second_payload.focus_refs == ("fact:intermediate",)
    assert second_payload.body["nodes"][0]["node_id"] == "fact:intermediate"


def test_item_prompt_requires_terminal_fact_and_focus_refs():
    assert "focus_refs" in GRAPH_ITEM_PLANNER_GUIDANCE
    assert "terminal" in GRAPH_ITEM_PLANNER_GUIDANCE
    assert "exactly one" in GRAPH_ITEM_PLANNER_GUIDANCE
