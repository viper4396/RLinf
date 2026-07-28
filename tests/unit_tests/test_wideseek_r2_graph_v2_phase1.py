# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Independent Phase 1 contract tests for the existing ``mas_graph`` path."""

import asyncio

import pytest

from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionResultProposal,
    EdgeProposal,
    EvidenceProposal,
    GraphEventType,
    NodeProposal,
)
from rlinf.agents.wideseek_r2.graph_memory.state import GraphRuntime
from rlinf.agents.wideseek_r2.graph_memory.tools import get_graph_tools_description
from rlinf.agents.wideseek_r2.graph_memory.validator import (
    GraphValidationError,
    bootstrap_entities,
)
from rlinf.algorithms.toolcall_parsers import WideSeekR2GraphQwenToolCallParser


def _runtime() -> GraphRuntime:
    return GraphRuntime.bootstrap(
        question="Who leads Acme?",
        answer_type="item",
        config={"enabled": True, "schema_version": "v2"},
    )


async def _entity(runtime: GraphRuntime) -> str:
    result = await bootstrap_entities(
        runtime,
        [
            NodeProposal(
                "acme",
                "entity",
                "organization:acme",
                {"canonical_name": "Acme"},
            )
        ],
    )
    return result.delta.node_ids[0]


async def _worker_source_and_candidate(runtime: GraphRuntime, entity_id: str):
    action = runtime.create_action("Find the leader", focus_refs=(entity_id,))
    runtime.mark_action_running(action.action_id, owner_sub_traj=1)
    tool_result = runtime.record_tool_result(
        tool_name="access",
        action_id=action.action_id,
        sub_traj_id=1,
        result="Acme is led by Ada.",
        url="https://example.test/acme",
    )
    proposal = EvidenceProposal(
        action_id=action.action_id,
        base_version=runtime.version,
        nodes=(
            NodeProposal(
                "source",
                "source",
                "source:https://example.test/acme",
                {
                    "uri": "https://example.test/acme",
                    "locator": "paragraph 1",
                    "content_hash": tool_result.result_hash,
                },
                tool_result_refs=(tool_result.tool_result_id,),
            ),
            NodeProposal(
                "candidate",
                "candidate",
                "candidate:ada",
                {"raw_value": "Ada"},
                tool_result_refs=(tool_result.tool_result_id,),
            ),
        ),
        edges=(
            EdgeProposal("candidate", "OBSERVED_IN", "source"),
            EdgeProposal("candidate", "ABOUT", entity_id),
        ),
        action_result=ActionResultProposal(status="completed"),
        created_by_sub_traj=1,
        created_by_role="subagent",
        tool_result_refs=(tool_result.tool_result_id,),
    )
    result = await runtime.add_mem(proposal)
    return (
        runtime.evidence_graph.get_by_canonical(
            "source:https://example.test/acme", "source"
        ).node_id,
        runtime.evidence_graph.get_by_canonical("candidate:ada", "candidate").node_id,
        result,
    )


def test_v2_bootstrap_has_no_plan_gate_or_join_and_is_one_shot():
    async def run():
        runtime = _runtime()
        assert runtime.activation_dag.gates == {}
        assert runtime.activation_dag.joins == {}
        assert runtime.contract is None
        assert runtime.evidence_graph.nodes == {}
        assert "answer" not in runtime.task_context()
        assert "reward" not in runtime.task_context()

        entity_id = await _entity(runtime)
        assert runtime.bootstrap_entities == (entity_id,)
        assert runtime.evidence_graph.nodes[entity_id].proposed_by_role == "main"
        with pytest.raises(GraphValidationError, match="BOOTSTRAP_ALREADY_CALLED"):
            await bootstrap_entities(runtime, [])

    asyncio.run(run())


def test_phase1_ownership_provenance_and_atomic_rollback():
    async def run():
        runtime = _runtime()
        entity_id = await _entity(runtime)

        with pytest.raises(GraphValidationError, match="MAIN_NODE_FORBIDDEN"):
            await runtime.edit_mem(
                base_version=runtime.version,
                operations=[
                    {
                        "op": "add_node",
                        "node": {
                            "client_ref": "bad_source",
                            "kind": "source",
                            "canonical_key": "bad",
                            "payload": {},
                        },
                    }
                ],
            )
        assert runtime.evidence_graph.nodes == {
            entity_id: runtime.evidence_graph.nodes[entity_id]
        }

        action = runtime.create_action("search", focus_refs=(entity_id,))
        runtime.mark_action_running(action.action_id, owner_sub_traj=2)
        with pytest.raises(GraphValidationError, match="SUBAGENT_NODE_FORBIDDEN"):
            await runtime.add_mem(
                EvidenceProposal(
                    action_id=action.action_id,
                    base_version=runtime.version,
                    nodes=(NodeProposal("entity", "entity", "entity:bad", {}),),
                    created_by_sub_traj=2,
                    created_by_role="subagent",
                )
            )
        assert runtime.evidence_graph.get_by_canonical("entity:bad", "entity") is None

        await _worker_source_and_candidate(runtime, entity_id)
        assert runtime.summary()["evidence_counts"]["source"] == 1
        assert runtime.summary()["evidence_counts"]["candidate"] == 1

        action = runtime.create_action("bad provenance")
        runtime.mark_action_running(action.action_id, owner_sub_traj=3)
        other = runtime.record_tool_result(
            tool_name="access",
            action_id=action.action_id,
            sub_traj_id=4,
            result="wrong owner",
            url="https://example.test/wrong",
        )
        before = (
            runtime.version,
            len(runtime.evidence_graph.nodes),
            len(runtime.event_log),
        )
        with pytest.raises(GraphValidationError, match="TOOL_RESULT_OWNER_MISMATCH"):
            await runtime.add_mem(
                EvidenceProposal(
                    action_id=action.action_id,
                    base_version=runtime.version,
                    nodes=(
                        NodeProposal(
                            "source",
                            "source",
                            "source:wrong",
                            {
                                "uri": "https://example.test/wrong",
                                "locator": "p1",
                            },
                            tool_result_refs=(other.tool_result_id,),
                        ),
                    ),
                    created_by_sub_traj=3,
                    created_by_role="subagent",
                )
            )
        assert (
            runtime.version,
            len(runtime.evidence_graph.nodes),
            len(runtime.event_log),
        ) == before

    asyncio.run(run())


def test_event_log_tombstone_and_active_view():
    async def run():
        runtime = _runtime()
        entity_id = await _entity(runtime)
        result = await runtime.edit_mem(
            base_version=runtime.version,
            operations=[{"op": "retire_node", "ref": entity_id, "reason": "merged"}],
        )
        assert result.delta.node_ids == (entity_id,)
        assert (
            runtime.evidence_graph.get_by_canonical("organization:acme", "entity")
            is None
        )
        retired = runtime.evidence_graph.get_by_canonical(
            "organization:acme", "entity", active_only=False
        )
        assert retired is not None and retired.active is False
        assert any(
            event.event_type == GraphEventType.RETIRE_NODE
            and entity_id in event.node_ids
            for event in runtime.event_log
        )
        assert runtime.active_graph.nodes == {}

    asyncio.run(run())


def test_source_hash_mismatch_is_rejected_atomically():
    async def run():
        runtime = _runtime()
        action = runtime.create_action("validate source")
        runtime.mark_action_running(action.action_id, owner_sub_traj=1)
        tool_result = runtime.record_tool_result(
            tool_name="access",
            action_id=action.action_id,
            sub_traj_id=1,
            result="authoritative text",
            url="https://example.test/source",
        )
        before = (runtime.version, len(runtime.evidence_graph.nodes))
        with pytest.raises(GraphValidationError, match="SOURCE_HASH_MISMATCH"):
            await runtime.add_mem(
                EvidenceProposal(
                    action_id=action.action_id,
                    base_version=runtime.version,
                    nodes=(
                        NodeProposal(
                            "source",
                            "source",
                            "source:bad-hash",
                            {
                                "uri": "https://example.test/source",
                                "locator": "paragraph 1",
                                "content_hash": "not-the-result-hash",
                            },
                            tool_result_refs=(tool_result.tool_result_id,),
                        ),
                    ),
                    created_by_sub_traj=1,
                    created_by_role="subagent",
                )
            )
        assert (runtime.version, len(runtime.evidence_graph.nodes)) == before

    asyncio.run(run())


def test_potential_conflict_queue_and_atomic_resolution():
    async def run():
        runtime = _runtime()
        entity_id = await _entity(runtime)
        result = await runtime.edit_mem(
            base_version=runtime.version,
            operations=[
                {
                    "op": "add_node",
                    "node": {
                        "client_ref": "claim_a",
                        "kind": "claim",
                        "canonical_key": "claim:leader:ada",
                        "payload": {
                            "subject_ref": entity_id,
                            "predicate": "leader",
                            "object": "Ada",
                        },
                    },
                },
                {
                    "op": "add_node",
                    "node": {
                        "client_ref": "claim_b",
                        "kind": "claim",
                        "canonical_key": "claim:leader:bob",
                        "payload": {
                            "subject_ref": entity_id,
                            "predicate": "leader",
                            "object": "Bob",
                        },
                    },
                },
            ],
        )
        claim_a, claim_b = result.delta.node_ids
        assert runtime.pending_conflict_ids

        conflict_result = await runtime.edit_mem(
            base_version=runtime.version,
            operations=[
                {
                    "op": "add_node",
                    "node": {
                        "client_ref": "conflict",
                        "kind": "conflict",
                        "canonical_key": "conflict:leader",
                        "payload": {
                            "competing_refs": [claim_a, claim_b],
                            "conflict_type": "mutually_exclusive_value",
                        },
                    },
                },
                {
                    "op": "add_edge",
                    "source_ref": "conflict",
                    "relation": "CONTAINS",
                    "target_ref": claim_a,
                },
                {
                    "op": "add_edge",
                    "source_ref": "conflict",
                    "relation": "CONTAINS",
                    "target_ref": claim_b,
                },
            ],
        )
        conflict_id = conflict_result.delta.conflict_ids[0]
        with pytest.raises(
            GraphValidationError, match="CONFLICT_RESOLUTION_INCOMPLETE"
        ):
            await runtime.edit_mem(
                base_version=runtime.version,
                operations=[
                    {
                        "op": "resolve_conflict",
                        "conflict_ref": conflict_id,
                        "winner_ref": claim_a,
                        "retire_refs": [],
                    }
                ],
            )

        result = await runtime.edit_mem(
            base_version=runtime.version,
            operations=[
                {
                    "op": "resolve_conflict",
                    "conflict_ref": conflict_id,
                    "winner_ref": claim_a,
                    "retire_refs": [claim_b],
                }
            ],
        )
        assert result.graph_version == runtime.version
        assert runtime.evidence_graph.nodes[claim_b].active is False
        assert conflict_id not in runtime.active_graph.nodes
        assert runtime.pending_conflict_ids.isdisjoint({conflict_id})
        assert any(
            event.event_type == GraphEventType.RESOLVE_CONFLICT
            and conflict_id in event.node_ids
            for event in runtime.event_log
        )

    asyncio.run(run())


def test_phase1_claim_queue_and_next_turn_promotion():
    async def run():
        runtime = _runtime()
        entity_id = await _entity(runtime)
        source_id, candidate_id, _ = await _worker_source_and_candidate(
            runtime, entity_id
        )
        claim_result = await runtime.edit_mem(
            base_version=runtime.version,
            operations=[
                {
                    "op": "add_node",
                    "node": {
                        "client_ref": "claim",
                        "kind": "claim",
                        "canonical_key": "claim:leader",
                        "payload": {
                            "subject_ref": entity_id,
                            "predicate": "leader",
                            "object": "Ada",
                        },
                    },
                },
                {
                    "op": "add_edge",
                    "source_ref": "claim",
                    "relation": "SUPPORTED_BY",
                    "target_ref": candidate_id,
                },
            ],
        )
        claim_id = claim_result.delta.node_ids[0]
        assert runtime.pending_claim_ids == {claim_id}
        with pytest.raises(GraphValidationError, match="SAME_TURN_PROMOTION"):
            await runtime.edit_mem(
                base_version=runtime.version,
                operations=[
                    {
                        "op": "promote_claim",
                        "claim_ref": claim_id,
                        "source_refs": [source_id],
                    }
                ],
            )
        runtime.begin_turn(runtime.main_turn + 1, "planner")
        result = await runtime.edit_mem(
            base_version=runtime.version,
            operations=[
                {
                    "op": "promote_claim",
                    "claim_ref": claim_id,
                    "source_refs": [source_id],
                    "fact": {"object": "Ada"},
                }
            ],
        )
        assert result.delta.fact_ids
        assert runtime.pending_claim_ids == set()

    asyncio.run(run())


def test_v2_parser_and_tool_schemas_enforce_modes():
    planner = {
        tool["function"]["name"] for tool in get_graph_tools_description("planner")
    }
    worker = {
        tool["function"]["name"] for tool in get_graph_tools_description("worker")
    }
    assert planner == {"call_sub", "read_mem", "edit_mem"}
    assert worker == {"add_mem"}

    async def run():
        parser = WideSeekR2GraphQwenToolCallParser()
        _, requests = await parser(
            '<tool_call>{"name":"call_sub","arguments":{"subtasks":[{"subtask":"find it","focus_refs":["e:1"]}]}}</tool_call>',
            role="planner",
        )
        assert requests[0].name == "subtask"
        assert "action_id" not in requests[0].arguments
        assert requests[0].arguments["focus_refs"] == ["e:1"]

        _, requests = await parser(
            '<tool_call>{"name":"call_sub","arguments":{"subtasks":[{"subtask":"find it"}]}}</tool_call>'
            '<tool_call>{"name":"read_mem","arguments":{}}</tool_call>',
            role="planner",
        )
        assert requests == []
        assert "MIXED_TOOL_MODE" in parser.last_error

        _, requests = await parser(
            '<tool_call>{"name":"add_mem","arguments":{"base_version":0,"nodes":[],"edges":[]}}</tool_call>'
            '<tool_call>{"name":"search","arguments":{"queries":[{"query":"Ada"}]}}</tool_call>',
            role="worker",
        )
        assert requests == []
        assert "MIXED_TOOL_MODE" in parser.last_error

        _, requests = await parser(
            '<tool_call>{"name":"read_mem","arguments":{}}</tool_call>',
            role="worker",
        )
        assert requests == []
        assert parser.last_error.startswith("Unknown worker")

    asyncio.run(run())
