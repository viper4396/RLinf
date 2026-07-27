# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

import asyncio

import pytest

from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionNode,
    ActionResultProposal,
    ActionState,
    ActivationEdge,
    EdgeProposal,
    EvidenceProposal,
    GateNode,
    NodeProposal,
    PayloadNode,
    TaskContract,
    TaskPlanProposal,
)
from rlinf.agents.wideseek_r2.graph_memory.state import GraphRuntime
from rlinf.agents.wideseek_r2.graph_memory.validator import (
    GraphValidationError,
    commit_evidence,
    compile_task_plan,
)


def _two_hop_plan() -> TaskPlanProposal:
    return TaskPlanProposal(
        contract=TaskContract(
            contract_id="contract:item",
            question="Which country?",
            answer_kind="item",
        ),
        actions=(
            ActionNode("action:hop_1", "Resolve the intermediate entity"),
            ActionNode("action:hop_2", "Resolve the terminal country"),
        ),
        gates=(
            GateNode("gate:hop_1", {"op": "true"}),
            GateNode(
                "gate:hop_2",
                {"op": "fact_exists", "canonical_key": "claim:country:verified"},
            ),
        ),
        payloads=(
            PayloadNode(
                "payload:hop_2",
                selector={"refs_from_gate": "gate:hop_2"},
                required=True,
            ),
        ),
        edges=(
            ActivationEdge("edge:hop_1", "gate:hop_1", "action:hop_1", "guards"),
            ActivationEdge("edge:hop_2", "gate:hop_2", "action:hop_2", "guards"),
            ActivationEdge(
                "edge:payload:hop_2",
                "payload:hop_2",
                "action:hop_2",
                "delivers",
            ),
            ActivationEdge(
                "edge:dependency", "action:hop_1", "action:hop_2", "precedes"
            ),
        ),
    )


def _evidence_proposal(runtime: GraphRuntime, action_id: str = "action:hop_1"):
    return EvidenceProposal(
        action_id=action_id,
        base_version=runtime.version,
        created_by_sub_traj=1,
        nodes=(
            NodeProposal(
                "entity",
                "entity",
                "person:target",
                {"entity_type": "person", "canonical_name": "Target"},
            ),
            NodeProposal(
                "source",
                "source",
                "url:country",
                {"uri": "https://example.test/country", "locator": "paragraph 1"},
            ),
            NodeProposal(
                "claim",
                "claim",
                "claim:country",
                {
                    "subject_ref": "entity",
                    "predicate": "country",
                    "object": "Chile",
                    "terminal": True,
                },
            ),
        ),
        edges=(
            EdgeProposal("source", "SUPPORTS", "claim"),
            EdgeProposal("claim", "ABOUT", "entity"),
        ),
        action_result=ActionResultProposal(status="completed"),
    )


def test_bootstrap_is_empty_and_ground_truth_free():
    runtime = GraphRuntime.bootstrap(
        question="Which country?", answer_type="item", config={"enabled": True}
    )

    assert runtime.version == 0
    assert runtime.evidence_graph.nodes == {}
    assert "answer" not in runtime.task_context()
    assert "reward" not in runtime.task_context()
    assert runtime.activation_dag.actions["action:plan_task"].state == ActionState.READY


def test_item_two_hop_activation_happens_once():
    async def run():
        runtime = GraphRuntime.bootstrap(
            question="Which country?", answer_type="item", config={"enabled": True}
        )
        await compile_task_plan(runtime, _two_hop_plan())

        assert runtime.activation_dag.actions["action:hop_1"].state == ActionState.READY
        assert (
            runtime.activation_dag.actions["action:hop_2"].state == ActionState.DORMANT
        )
        assert len(runtime.pending_events("action:hop_1")) == 1

        runtime.mark_action_running("action:hop_1")
        result = await commit_evidence(runtime, _evidence_proposal(runtime))

        assert result.delta.fact_ids
        assert runtime.activation_dag.gates["gate:hop_2"].satisfied is True
        assert runtime.activation_dag.actions["action:hop_2"].state == ActionState.READY
        assert len(runtime.pending_events("action:hop_2")) == 1
        runtime.evaluate_activation()
        assert len(runtime.pending_events("action:hop_2")) == 1

        event = runtime.pending_events("action:hop_2", consume=True)[0]
        assert event.graph_version == runtime.version
        assert event.allowed_reads
        assert runtime.read_evidence(
            list(event.allowed_reads), action_id="action:hop_2"
        )

    asyncio.run(run())


def test_stale_new_proposal_is_rejected_atomically():
    async def run():
        runtime = GraphRuntime.bootstrap(
            question="Which country?", answer_type="item", config={"enabled": True}
        )
        await compile_task_plan(runtime, _two_hop_plan())
        runtime.mark_action_running("action:hop_1")
        proposal = _evidence_proposal(runtime)
        await commit_evidence(runtime, proposal)
        node_count = len(runtime.evidence_graph.nodes)

        with pytest.raises(GraphValidationError, match="VERSION_CONFLICT"):
            await commit_evidence(
                runtime,
                EvidenceProposal(
                    **{
                        **proposal.__dict__,
                        "proposal_id": "stale",
                        "action_id": "action:hop_2",
                        "nodes": proposal.nodes
                        + (
                            NodeProposal(
                                "new",
                                "candidate",
                                "candidate:new",
                                {"raw_value": "new"},
                            ),
                        ),
                    }
                ),
            )
        assert len(runtime.evidence_graph.nodes) == node_count

    asyncio.run(run())


def test_contextvar_runtime_isolation():
    async def read(runtime, value):
        token = runtime.context_token()
        try:
            from rlinf.agents.wideseek_r2.graph_memory.state import get_graph_runtime

            await asyncio.sleep(0)
            return get_graph_runtime().question, value
        finally:
            runtime.reset_context(token)

    async def run():
        first = GraphRuntime.bootstrap(question="one", answer_type="item")
        second = GraphRuntime.bootstrap(question="two", answer_type="item")
        assert await asyncio.gather(read(first, 1), read(second, 2)) == [
            ("one", 1),
            ("two", 2),
        ]

    asyncio.run(run())


def test_activation_cycle_is_rejected_before_install():
    async def run():
        runtime = GraphRuntime.bootstrap(
            question="Which country?", answer_type="item", config={"enabled": True}
        )
        plan = TaskPlanProposal(
            contract=TaskContract("contract:cycle", "Which country?", "item"),
            actions=(ActionNode("a", "a"), ActionNode("b", "b")),
            edges=(
                ActivationEdge("ab", "a", "b", "precedes"),
                ActivationEdge("ba", "b", "a", "precedes"),
            ),
        )
        with pytest.raises(GraphValidationError, match="DAG_CYCLE"):
            await compile_task_plan(runtime, plan)
        assert runtime.contract is None

    asyncio.run(run())
