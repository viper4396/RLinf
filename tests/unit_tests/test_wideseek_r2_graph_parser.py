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
                "create_sub_agents",
                {
                    "sub_agents": [
                        {
                            "action_id": "action:hop_1",
                            "prompt": "Find the intermediate entity",
                            "input_refs": ["entity:root"],
                            "expected_output": {"node_kinds": ["claim"]},
                        }
                    ]
                },
            ),
            role="planner",
        )
        assert requests[0].name == "subtask"
        assert requests[0].arguments["action_id"] == "action:hop_1"
        assert requests[0].arguments["input_refs"] == ["entity:root"]
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
            "read_evidence",
            {"refs": ["fact:one"]},
        ) + _call(
            "search",
            {"queries": [{"query": "country"}]},
        )
        _, requests = await parser(response, role="worker")
        assert requests == []
        assert parser.last_error == "Mixed graph/external tool phases are not allowed"

    asyncio.run(run())


def test_graph_parser_reports_incomplete_tags():
    async def run():
        parser = WideSeekR2GraphQwenToolCallParser()
        _, requests = await parser(
            'prefix <tool_call>{"name": "read_evidence"}', role="worker"
        )
        assert requests == []
        assert parser.last_error == "Tool call tags were incomplete"

    asyncio.run(run())
