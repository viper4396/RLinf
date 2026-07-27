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

import json
import logging
import re

import regex

from rlinf.algorithms.registry import register_toolcall_parser
from rlinf.data.tool_call.tool_io_struct import (
    ToolRequest,
    ToolResponse,
)


@register_toolcall_parser("qwen2.5")
class Qwen25ToolCallParser:
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self):
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(
            r"<tool_call>(.*?)</tool_call>", regex.DOTALL
        )

    async def __call__(self, responses_text: str) -> tuple[str, list[ToolRequest]]:
        text = responses_text
        if (
            self.tool_call_start_token not in text
            or self.tool_call_end_token not in text
        ):
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(
                    ToolRequest(
                        name=name, arguments=json.dumps(arguments, ensure_ascii=False)
                    )
                )
            except Exception as e:
                logging.error(f"Failed to decode tool call: {e}")

        # remaing text exclude tool call tokens
        content = self.tool_call_regex.sub("", text)

        return content, function_calls


@register_toolcall_parser("searchr1-qwen")
class Searchr1QwenToolCallParser:
    def __init__(self) -> None:
        self.tool_call_start_token: str = "<search>"
        self.tool_call_end_token: str = "</search>"
        self.tool_call_regex = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        self.repairable_tool_call_regex = re.compile(
            r"<search(?:\s+[^>]*)?>(.*?)(?:</search>|(?=<answer>)|$)",
            re.DOTALL,
        )

    async def __call__(self, response_text: str) -> tuple[str, list[ToolRequest]]:
        matches = self.tool_call_regex.findall(response_text)
        parser = self.tool_call_regex
        if not matches:
            # Recover common model slips such as ``<search query>...`` and a
            # missing closing tag. The agent loop still records these calls as
            # format-invalid so training/evaluation can distinguish repairs.
            matches = self.repairable_tool_call_regex.findall(response_text)
            parser = self.repairable_tool_call_regex
        function_calls = []
        if matches:
            match = matches[-1].strip()
            if match and "<" not in match and ">" not in match:
                function_calls.append(
                    ToolRequest(name="search", arguments={"keyword": match})
                )

        # remaining text exclude tool call tokens
        content = parser.sub("", response_text)

        return content, function_calls


@register_toolcall_parser("rstar2-qwen")
class Rstar2QwenToolCallParser:
    def __init__(self) -> None:
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    async def __call__(
        self, response_text: str
    ) -> tuple[str, list[ToolRequest | ToolResponse]]:
        if (
            self.tool_call_start_token not in response_text
            or self.tool_call_end_token not in response_text
        ):
            return response_text, []

        matches = self.tool_call_regex.findall(response_text)
        return_function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                return_function_calls.append(
                    ToolRequest(name=name, arguments=arguments)
                )
            except Exception as e:
                return_function_calls.append(
                    ToolResponse(text=f"Failed to decode tool call: {e}")
                )

        return response_text, return_function_calls


@register_toolcall_parser("wideseek_r1-qwen")
class WideSeekQwenToolCallParser:
    """Tool-call parser for WideSeek-R1 planner/worker/single-agent roles."""

    def __init__(self) -> None:
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    @staticmethod
    def _parse_planner_calls(
        tool_name: str,
        tool_arguments: dict,
        max_workers_per_planner: int,
    ) -> list[ToolRequest]:
        if tool_name != "create_sub_agents":
            return []
        sub_agents = tool_arguments.get("sub_agents", [])
        if not isinstance(sub_agents, list):
            return []

        function_calls = []
        for sub_agent in sub_agents[:max_workers_per_planner]:
            if not isinstance(sub_agent, dict):
                continue
            prompt = sub_agent.get("prompt", "")
            if not isinstance(prompt, str) or not prompt:
                continue
            function_calls.append(
                ToolRequest(name="subtask", arguments={"subtask": prompt})
            )
        return function_calls

    @staticmethod
    def _parse_worker_calls(
        tool_name: str,
        tool_arguments: dict,
        max_toolcall_per_worker: int,
    ) -> list[ToolRequest]:
        function_calls = []
        if tool_name == "search":
            searches = tool_arguments.get("queries", [])
            if not isinstance(searches, list):
                return []
            for search_item in searches[:max_toolcall_per_worker]:
                if not isinstance(search_item, dict):
                    continue
                query = search_item.get("query", "")
                if not isinstance(query, str) or not query:
                    continue
                topk = search_item.get("count", None)
                if topk:
                    function_calls.append(
                        ToolRequest(
                            name="search",
                            arguments={"query": query, "topk": topk},
                        )
                    )
                else:
                    function_calls.append(
                        ToolRequest(name="search", arguments={"query": query})
                    )

        elif tool_name == "access":
            accesses = tool_arguments.get("urls", [])
            if not isinstance(accesses, list):
                return []
            for access_item in accesses[:max_toolcall_per_worker]:
                if not isinstance(access_item, dict):
                    continue
                url = access_item.get("url", "")
                if not isinstance(url, str) or not url:
                    continue
                info_to_extract = access_item.get("info_to_extract", None)
                function_calls.append(
                    ToolRequest(
                        name="access",
                        arguments={
                            "url": url,
                            "access_token": 25000,
                            "info_to_extract": info_to_extract,
                        },
                    )
                )
        return function_calls

    @staticmethod
    def _parse_single_calls(tool_name: str, tool_arguments: dict) -> list[ToolRequest]:
        if tool_name == "search":
            query = tool_arguments.get("query", "")
            if not isinstance(query, str) or not query:
                return []
            topk = tool_arguments.get("count", None)
            if topk:
                return [
                    ToolRequest(
                        name="search",
                        arguments={"query": query, "topk": topk},
                    )
                ]
            return [ToolRequest(name="search", arguments={"query": query})]

        if tool_name == "access":
            url = tool_arguments.get("url", "")
            if not isinstance(url, str) or not url:
                return []
            info_to_extract = tool_arguments.get("info_to_extract", None)
            return [
                ToolRequest(
                    name="access",
                    arguments={
                        "url": url,
                        "access_token": 25000,
                        "info_to_extract": info_to_extract,
                    },
                )
            ]
        return []

    async def __call__(
        self,
        response_text: str,
        *,
        role: str,
        max_workers_per_planner: int = 10,
        max_toolcall_per_worker: int = 5,
    ) -> tuple[str, list[ToolRequest]]:
        if (
            self.tool_call_start_token not in response_text
            or self.tool_call_end_token not in response_text
        ):
            return response_text, []

        matches = self.tool_call_regex.findall(response_text)
        if not matches:
            return response_text, []

        try:
            tool_call_json = json.loads(matches[0].strip())
        except Exception:
            return response_text, []

        if not isinstance(tool_call_json, dict):
            return response_text, []
        tool_name = tool_call_json.get("name")
        tool_arguments = tool_call_json.get("arguments", {})
        if not isinstance(tool_arguments, dict):
            return response_text, []

        if role == "planner":
            function_calls = self._parse_planner_calls(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                max_workers_per_planner=max_workers_per_planner,
            )
        elif role == "worker":
            function_calls = self._parse_worker_calls(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                max_toolcall_per_worker=max_toolcall_per_worker,
            )
        elif role == "single":
            function_calls = self._parse_single_calls(
                tool_name=tool_name, tool_arguments=tool_arguments
            )
        else:
            function_calls = []

        # remaining text exclude tool call tokens
        content = self.tool_call_regex.sub("", response_text)
        return content, function_calls


@register_toolcall_parser("wideseek_r2-qwen")
class WideSeekR2QwenToolCallParser(WideSeekQwenToolCallParser):
    """Tool-call parser for WideSeek-R2 planner/worker/single-agent roles.

    WideSeek-R2 makes the per-turn sub-agent cap optional: a negative
    ``max_workers_per_planner`` means a planner may create an unlimited number
    of sub-agents in a single ``create_sub_agents`` call, while a non-negative
    value caps the count exactly like ``wideseek_r1-qwen``. Only
    ``_parse_planner_calls`` is overridden so the shared base ``__call__`` (and
    ``wideseek_r1-qwen``) stay untouched.
    """

    @staticmethod
    def _parse_planner_calls(
        tool_name: str,
        tool_arguments: dict,
        max_workers_per_planner: int,
    ) -> list[ToolRequest]:
        if tool_name != "create_sub_agents":
            return []
        sub_agents = tool_arguments.get("sub_agents", [])
        if not isinstance(sub_agents, list):
            return []

        # Negative -> unlimited; non-negative -> cap at that many sub-agents.
        # (Note: `sub_agents[:max_workers_per_planner]` cannot express "no cap"
        # since a negative index would drop trailing items, so branch instead.)
        if max_workers_per_planner >= 0:
            sub_agents = sub_agents[:max_workers_per_planner]

        function_calls = []
        for sub_agent in sub_agents:
            if not isinstance(sub_agent, dict):
                continue
            prompt = sub_agent.get("prompt", "")
            if not isinstance(prompt, str) or not prompt:
                continue
            function_calls.append(
                ToolRequest(name="subtask", arguments={"subtask": prompt})
            )
        return function_calls


@register_toolcall_parser("wideseek_r2-graph-qwen")
class WideSeekR2GraphQwenToolCallParser(WideSeekR2QwenToolCallParser):
    """Structured parser for the Phase 1 ``mas_graph`` workflow.

    Unlike the legacy parser, planner entries retain their action scope and
    worker-local graph tools are parsed as first-class requests.  Invalid or
    mixed phases are surfaced through ``last_error`` instead of silently
    becoming an empty tool-call list.
    """

    planner_tools = {
        "submit_task_plan",
        "create_sub_agents",
        "read_graph_summary",
        "propose_finish",
        "propose_plan_patch",
    }
    worker_tools = {
        "search",
        "access",
        "read_evidence",
        "submit_evidence",
        "report_action_status",
        "propose_next_actions",
    }
    graph_tools = {
        "submit_task_plan",
        "read_graph_summary",
        "propose_finish",
        "propose_plan_patch",
        "read_evidence",
        "submit_evidence",
        "report_action_status",
        "propose_next_actions",
    }

    def __init__(self) -> None:
        super().__init__()
        self.last_error: str | None = None

    def _error(self, message: str) -> None:
        self.last_error = message

    def _parse_graph_planner_calls(
        self, tool_name: str, tool_arguments: dict, max_workers_per_planner: int
    ) -> list[ToolRequest]:
        if tool_name == "create_sub_agents":
            sub_agents = tool_arguments.get("sub_agents", [])
            if not isinstance(sub_agents, list):
                self._error("create_sub_agents.sub_agents must be a list")
                return []
            if max_workers_per_planner >= 0:
                sub_agents = sub_agents[:max_workers_per_planner]
            requests = []
            for index, sub_agent in enumerate(sub_agents):
                if not isinstance(sub_agent, dict):
                    self._error(f"sub_agents[{index}] must be an object")
                    continue
                prompt = sub_agent.get("prompt", "")
                action_id = sub_agent.get("action_id", "")
                if not isinstance(prompt, str) or not prompt:
                    self._error(f"sub_agents[{index}].prompt is required")
                    continue
                if not isinstance(action_id, str) or not action_id:
                    self._error(f"sub_agents[{index}].action_id is required")
                    continue
                requests.append(
                    ToolRequest(
                        name="subtask",
                        arguments={
                            "subtask": prompt,
                            "action_id": action_id,
                            "input_refs": sub_agent.get("input_refs", []),
                            "expected_output": sub_agent.get("expected_output", {}),
                        },
                    )
                )
            return requests
        if tool_name not in self.planner_tools:
            self._error(f"Unknown planner graph tool: {tool_name!r}")
            return []
        if not isinstance(tool_arguments, dict):
            self._error(f"Arguments for {tool_name!r} must be an object")
            return []
        return [ToolRequest(name=tool_name, arguments=tool_arguments)]

    def _parse_graph_worker_calls(
        self, tool_name: str, tool_arguments: dict, max_toolcall_per_worker: int
    ) -> list[ToolRequest]:
        if tool_name in {
            "read_evidence",
            "submit_evidence",
            "report_action_status",
            "propose_next_actions",
        }:
            if not isinstance(tool_arguments, dict):
                self._error(f"Arguments for {tool_name!r} must be an object")
                return []
            return [ToolRequest(name=tool_name, arguments=tool_arguments)]
        if tool_name in {"search", "access"}:
            requests = super()._parse_worker_calls(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                max_toolcall_per_worker=max_toolcall_per_worker,
            )
            if not requests:
                self._error(f"Malformed {tool_name} arguments")
            return requests
        self._error(f"Unknown worker graph tool: {tool_name!r}")
        return []

    async def __call__(
        self,
        response_text: str,
        *,
        role: str,
        max_workers_per_planner: int = 10,
        max_toolcall_per_worker: int = 5,
    ) -> tuple[str, list[ToolRequest]]:
        self.last_error = None
        has_start = self.tool_call_start_token in response_text
        has_end = self.tool_call_end_token in response_text
        if not has_start and not has_end:
            return response_text, []
        if has_start != has_end:
            self._error("Tool call tags were incomplete")
            return response_text, []
        matches = self.tool_call_regex.findall(response_text)
        if not matches:
            self._error("Tool call tags were present but no complete call was found")
            return response_text, []
        requests: list[ToolRequest] = []
        for match in matches:
            try:
                tool_call_json = json.loads(match.strip())
            except Exception as exc:
                self._error(f"Invalid tool-call JSON: {exc}")
                return self.tool_call_regex.sub("", response_text), []
            if not isinstance(tool_call_json, dict):
                self._error("Tool call must be an object")
                return self.tool_call_regex.sub("", response_text), []
            tool_name = tool_call_json.get("name")
            tool_arguments = tool_call_json.get("arguments", {})
            if not isinstance(tool_name, str) or not isinstance(tool_arguments, dict):
                self._error("Tool call requires string name and object arguments")
                return self.tool_call_regex.sub("", response_text), []
            if role == "planner":
                parsed = self._parse_graph_planner_calls(
                    tool_name, tool_arguments, max_workers_per_planner
                )
            elif role == "worker":
                parsed = self._parse_graph_worker_calls(
                    tool_name, tool_arguments, max_toolcall_per_worker
                )
            else:
                self._error(f"mas_graph does not support role {role!r}")
                parsed = []
            if self.last_error:
                return self.tool_call_regex.sub("", response_text), []
            requests.extend(parsed)

        # Apply the per-turn caps after aggregating multiple tags.  The legacy
        # parser only received one call at a time, so applying the cap here is
        # necessary to prevent a model from bypassing it by emitting several
        # adjacent tool-call blocks.
        if role == "planner" and max_workers_per_planner >= 0:
            requests = requests[:max_workers_per_planner]
        elif role == "worker":
            requests = requests[:max_toolcall_per_worker]
        if requests:
            phases = {
                "graph" if request.name in self.graph_tools else "external"
                for request in requests
                if request.name != "subtask"
            }
            if len(phases) > 1:
                self._error("Mixed graph/external tool phases are not allowed")
                requests = []
        content = self.tool_call_regex.sub("", response_text)
        return content, requests
