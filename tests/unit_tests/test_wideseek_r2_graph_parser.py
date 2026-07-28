# Copyright 2026 The RLinf Authors.

import asyncio
import json

from rlinf.algorithms.toolcall_parsers import WideSeekR2GraphQwenToolCallParser


def _call(name, arguments):
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
    )


def test_graph_parser_preserves_structured_action_scope():
    async def run():
        parser = WideSeekR2GraphQwenToolCallParser()
        _, requests = await parser(
            _call(
                "call_sub",
                {
                    "subtasks": [
                        {
                            "subtask": "Find the intermediate entity",
                            "focus_refs": ["entity:root"],
                            "output_contract": {"node_kinds": ["candidate"]},
                        }
                    ]
                },
            ),
            role="planner",
        )
        assert requests[0].name == "subtask"
        assert "action_id" not in requests[0].arguments
        assert requests[0].arguments["focus_refs"] == ["entity:root"]
        assert parser.last_error is None

    asyncio.run(run())


def test_graph_parser_rejects_unknown_tool_instead_of_silently_dropping():
    async def run():
        parser = WideSeekR2GraphQwenToolCallParser()
        _, requests = await parser(_call("delete_memory", {}), role="worker")
        assert requests == []
        assert parser.last_error and "Unknown worker" in parser.last_error

    asyncio.run(run())


def test_graph_parser_rejects_mixed_multi_tag_phases():
    async def run():
        parser = WideSeekR2GraphQwenToolCallParser()
        response = _call(
            "add_mem",
            {"base_version": 0, "nodes": [], "edges": []},
        ) + _call(
            "search",
            {"queries": [{"query": "country"}]},
        )
        _, requests = await parser(response, role="worker")
        assert requests == []
        assert parser.last_error.startswith("MIXED_TOOL_MODE:")

    asyncio.run(run())


def test_graph_parser_reports_incomplete_tags():
    async def run():
        parser = WideSeekR2GraphQwenToolCallParser()
        _, requests = await parser(
            'prefix <tool_call>{"name": "add_mem"}', role="worker"
        )
        assert requests == []
        assert parser.last_error == "Tool call tags were incomplete"

    asyncio.run(run())
