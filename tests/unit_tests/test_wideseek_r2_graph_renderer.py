# Copyright 2026 The RLinf Authors.

import asyncio

import pytest

from rlinf.agents.wideseek_r2.graph_memory.renderer import (
    RenderError,
    audit_item,
    render,
)
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionNode,
    EdgeProposal,
    EvidenceProposal,
    NodeProposal,
    TaskContract,
    TaskPlanProposal,
)
from rlinf.agents.wideseek_r2.graph_memory.state import GraphRuntime
from rlinf.agents.wideseek_r2.graph_memory.validator import (
    commit_evidence,
    compile_task_plan,
)


def test_renderer_blocks_direct_answer_without_accepted_fact():
    async def run():
        runtime = GraphRuntime.bootstrap(question="q", answer_type="item")
        await compile_task_plan(
            runtime,
            TaskPlanProposal(
                contract=TaskContract("contract", "q", "item"),
                actions=(ActionNode("lookup", "lookup"),),
            ),
        )
        assert not audit_item(runtime).passed
        with pytest.raises(RenderError):
            render(runtime)

    asyncio.run(run())


def test_renderer_uses_only_verified_fact():
    async def run():
        runtime = GraphRuntime.bootstrap(question="q", answer_type="item")
        await compile_task_plan(
            runtime,
            TaskPlanProposal(
                contract=TaskContract("contract", "q", "item"),
                actions=(ActionNode("lookup", "lookup"),),
            ),
        )
        runtime.mark_action_running("lookup")
        proposal = EvidenceProposal(
            action_id="lookup",
            base_version=runtime.version,
            nodes=(
                NodeProposal("entity", "entity", "entity:q", {"canonical_name": "Q"}),
                NodeProposal(
                    "source", "source", "url:q", {"uri": "https://q", "locator": "p1"}
                ),
                NodeProposal(
                    "claim",
                    "claim",
                    "claim:q",
                    {
                        "subject_ref": "entity",
                        "predicate": "answer",
                        "object": "A",
                        "terminal": True,
                    },
                ),
            ),
            edges=(EdgeProposal("source", "SUPPORTS", "claim"),),
        )
        await commit_evidence(runtime, proposal)
        assert render(runtime) == "```markdown\n| Item |\n| :--- |\n| A |\n```"

    asyncio.run(run())
