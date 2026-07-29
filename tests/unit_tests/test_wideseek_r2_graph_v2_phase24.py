# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Phase 2 retrieval and Phase 4 Audit/Render contract tests."""

import asyncio

from omegaconf import OmegaConf

from rlinf.agents.wideseek_r2.graph_memory.agent_loop import (
    WideSeekR2GraphAgentLoopWorker,
)
from rlinf.agents.wideseek_r2.graph_memory.audit import (
    parse_audit_pass,
    record_audit_outcome,
    start_audit,
    terminal_invariants,
)
from rlinf.agents.wideseek_r2.graph_memory.embedding_index import (
    DeterministicEmbeddingIndex,
)
from rlinf.agents.wideseek_r2.graph_memory.renderer import (
    record_render_outcome,
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


def _node(node_id, kind, key, payload, status=EvidenceStatus.ACTIVE):
    return EvidenceNode(
        node_id=node_id,
        kind=kind,
        canonical_key=key,
        payload=payload,
        status=status,
        created_at_version=0,
        updated_at_version=0,
    )


def _grounded_runtime() -> GraphRuntime:
    runtime = _runtime()
    graph = runtime.evidence_graph
    entity = _node(
        "entity:acme",
        EvidenceKind.ENTITY,
        "organization:acme",
        {"entity_type": "organization", "canonical_name": "Acme"},
    )
    source = _node(
        "source:acme",
        EvidenceKind.SOURCE,
        "source:https://example.test/acme",
        {
            "uri": "https://example.test/acme",
            "locator": "p1",
            "excerpt": "Ada leads Acme.",
        },
    )
    claim = _node(
        "claim:leader:ada",
        EvidenceKind.CLAIM,
        "claim:leader:ada",
        {"subject_ref": entity.node_id, "predicate": "leader", "object": "Ada"},
        EvidenceStatus.PROMOTED,
    )
    fact = _node(
        "fact:leader:ada",
        EvidenceKind.FACT,
        "fact:leader:ada",
        {
            "subject_ref": entity.node_id,
            "predicate": "leader",
            "object": "Ada",
            "value": "Ada",
            "terminal": True,
            "claim_ref": claim.node_id,
            "source_refs": [source.node_id],
        },
        EvidenceStatus.VERIFIED,
    )
    for node in (entity, source, claim, fact):
        graph.add_node(node)
    graph.add_edge(
        EvidenceEdge("edge:verified", claim.node_id, "VERIFIED_AS", fact.node_id)
    )
    graph.add_edge(
        EvidenceEdge("edge:source", fact.node_id, "SUPPORTED_BY", source.node_id)
    )
    return runtime


def test_phase2_index_and_action_payload_are_deterministic_and_scoped():
    runtime = _grounded_runtime()
    index = DeterministicEmbeddingIndex(64)
    index.rebuild(runtime.active_graph)
    assert set(index._vectors) == {"entity:acme", "fact:leader:ada"}
    first = index.search("Who leads Acme?\nCurrent subtask: verify leader", 2)
    second = index.search("Who leads Acme?\nCurrent subtask: verify leader", 2)
    assert first == second

    action = runtime.create_action(
        "Verify the leader", focus_refs=("claim:leader:ada",)
    )
    assert action.payload_ids
    payloads = [runtime.activation_dag.payloads[item] for item in action.payload_ids]
    assert payloads[0].graph_version == runtime.version
    body_kinds = {
        item["kind"] for payload in payloads for item in payload.body.get("nodes", [])
    }
    assert "claim" in body_kinds
    assert "conflict" not in body_kinds
    assert all(
        payload.token_count <= runtime.config.max_payload_tokens for payload in payloads
    )


def test_phase2_required_focus_budget_becomes_missing_context():
    runtime = _runtime(max_payload_nodes=1, max_payload_tokens=8)
    runtime.evidence_graph.add_node(
        _node(
            "entity:acme",
            EvidenceKind.ENTITY,
            "organization:acme",
            {"canonical_name": "Acme", "description": "A long description"},
        )
    )
    action = runtime.create_action("Find leader", focus_refs=("entity:acme",))
    assert action.state.value == "missing_context"
    assert action.metadata["missing_context"]


def test_phase4_audit_invariants_and_render_format_retry():
    runtime = _grounded_runtime()
    start_audit(runtime, "old response")
    report = record_audit_outcome(
        runtime,
        model_pass=parse_audit_pass('{"status":"AUDIT_PASS"}'),
        response_text='{"status":"AUDIT_PASS"}',
    )
    assert report.passed
    assert terminal_invariants(runtime)[0]

    payload = start_render(runtime)
    assert payload["allowed_refs"] == sorted(payload["allowed_refs"])
    assert runtime.render_payload_ids
    invalid = validate_render_answer(
        "```markdown\n| Item |\n| --- |\n| fact:outside |\n```",
        runtime,
    )
    assert not invalid.valid
    assert invalid.code == "OUTSIDE_PAYLOAD_REF"
    record_render_outcome(runtime, invalid)
    valid = validate_render_answer(
        "```markdown\n| Item |\n| --- |\n| Ada |\n```",
        runtime,
    )
    assert valid.valid
    record_render_outcome(runtime, valid)
    assert runtime.render_records[-1]["status"] == "passed"


def test_phase4_unresolved_claim_blocks_audit_and_payload_is_bounded():
    runtime = _runtime(max_source_excerpt_tokens=2)
    runtime.evidence_graph.add_node(
        _node(
            "claim:open",
            EvidenceKind.CLAIM,
            "claim:open",
            {"subject_ref": "entity:acme", "predicate": "leader", "object": "Ada"},
        )
    )
    start_audit(runtime)
    report = record_audit_outcome(runtime, model_pass=True, response_text="AUDIT_PASS")
    assert not report.passed
    assert "no_active_claim_without_fact" in report.missing
    assert runtime.audit_records[-1]["status"] == "INCOMPLETE"


def test_phase4_empty_graph_can_pass_mechanical_invariants():
    runtime = GraphRuntime.bootstrap(
        question="What is Acme?",
        answer_type="table",
        config={"enabled": True, "schema_version": "v2"},
    )
    start_audit(runtime)
    report = record_audit_outcome(runtime, model_pass=True, response_text="AUDIT_PASS")
    assert report.passed
    assert parse_audit_pass("AUDIT_PASS")


class _StubGraphWorker(WideSeekR2GraphAgentLoopWorker):
    """Backend-free planner stub for the normal -> Audit -> Render chain."""

    def __init__(self, outputs):
        self.cfg = OmegaConf.create(
            {
                "agentloop": {
                    "max_planner_turns": 2,
                    "max_workers_per_planner": -1,
                    "max_toolcall_per_worker": 5,
                }
            }
        )
        self.max_total_len = 1024
        self.return_logprobs = False
        self.outputs = list(outputs)
        self.tokenizer = self

    def decode(self, values):
        return values[0] if values else ""

    def _build_role_context(self, **_kwargs):
        return [1]

    async def _before_role_turn(self, *, prompt_ids, **_kwargs):
        return prompt_ids

    async def generate(self, *_args, **_kwargs):
        return {"output_ids": [self.outputs.pop(0)], "finish_reason": "stop"}

    async def extract_tool_calls(self, *_args, **_kwargs):
        return [], None

    def get_tool_response_ids(self, _messages):
        return []

    def release_affinity(self, _conversation_id):
        return None


def test_phase4_agent_loop_preserves_response_then_runs_audit_and_render():
    runtime = _grounded_runtime()
    worker = _StubGraphWorker(
        [
            "normal response",
            '{"status":"AUDIT_PASS"}',
            "```markdown\n| Item |\n| --- |\n| Ada |\n```",
        ]
    )
    token = runtime.context_token()
    try:
        _output, answer, _turns, _turn_idx = asyncio.run(
            worker._run_graph_planner_role("Who leads Acme?", 0, "item")
        )
    finally:
        runtime.reset_context(token)
    assert answer.startswith("```markdown")
    assert runtime.workflow_phase == "done"
    assert runtime.answer_source == "render_response"
    assert runtime.audit_attempt == 1
    assert runtime.render_attempt == 1


def test_item_strict_audit_does_not_fall_back_to_normal_response():
    runtime = _runtime(max_audit_attempts=1)
    worker = _StubGraphWorker(["unsupported normal response", "AUDIT_PASS"])
    token = runtime.context_token()
    try:
        _output, answer, _turns, _turn_idx = asyncio.run(
            worker._run_graph_planner_role("Who leads Acme?", 0, "item")
        )
    finally:
        runtime.reset_context(token)
    assert answer == ""
    assert runtime.answer_source == "audit_failed"
    assert runtime.terminal_failure == "AUDIT_REJECTED"
