# Copyright 2026 The RLinf Authors.

import asyncio

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
    assert planner == {"call_sub", "read_mem", "edit_mem"}
    assert worker == {"add_mem"}
    assert "read_mem" not in worker


def test_worker_read_mem_is_not_exposed_or_executable():
    async def run():
        runtime = GraphRuntime.bootstrap(
            question="q", answer_type="item", config={"enabled": True}
        )
        executor = GraphToolExecutor(runtime)
        response = await executor.execute(
            ToolRequest("read_mem", {"refs": []}),
            role="worker",
        )
        assert "GRAPH_TOOL_ERROR" in response.text

    asyncio.run(run())
