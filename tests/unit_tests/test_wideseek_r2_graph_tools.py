# Copyright 2026 The RLinf Authors.

import asyncio

from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionNode,
    ActionState,
    TaskContract,
    TaskPlanProposal,
)
from rlinf.agents.wideseek_r2.graph_memory.state import GraphRuntime
from rlinf.agents.wideseek_r2.graph_memory.tools import (
    GraphToolExecutor,
    get_graph_tools_description,
)
from rlinf.data.tool_call.tool_io_struct import ToolRequest


def test_graph_tool_schemas_are_role_scoped():
    planner = {
        tool["function"]["name"] for tool in get_graph_tools_description("planner")
    }
    worker = {
        tool["function"]["name"] for tool in get_graph_tools_description("worker")
    }
    assert "submit_task_plan" in planner
    assert "read_evidence" not in planner
    assert "read_evidence" in worker
    assert "submit_task_plan" not in worker


def test_read_evidence_enforces_action_acl():
    async def run():
        runtime = GraphRuntime.bootstrap(
            question="q", answer_type="item", config={"enabled": True}
        )
        await runtime.submit_task_plan(
            TaskPlanProposal(
                contract=TaskContract("contract", "q", "item"),
                actions=(ActionNode("lookup", "lookup", state=ActionState.DORMANT),),
            )
        )
        executor = GraphToolExecutor(runtime)
        response = await executor.execute(
            ToolRequest("read_evidence", {"refs": ["missing"]}),
            role="worker",
            action_id="lookup",
        )
        assert "GRAPH_TOOL_ERROR" in response.text

    asyncio.run(run())
