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
            r"(?:<search(?:\s+[^>]*)?>|=search>)(.*?)(?:</search>|(?=<answer>)|$)",
            re.DOTALL,
        )

    async def __call__(self, response_text: str) -> tuple[str, list[ToolRequest]]:
        matches = self.tool_call_regex.findall(response_text)
        parser = self.tool_call_regex
        if not matches:
            # Recover common model slips such as ``<search query>...``,
            # ``=search>...``, and a missing closing tag. The agent loop still
            # records these calls as format-invalid so evaluation can
            # distinguish repairs.
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
    """Structured parser for the v2 ``mas_graph`` workflow.

    The registration name is intentionally unchanged so existing configs keep
    using ``workflow: mas_graph``.  v2 changes the tool vocabulary and keeps
    mode validation in this parser, before any external or graph side effect.
    """

    planner_tools = {
        "call_sub",
        "read_mem",
        "edit_mem",
    }
    worker_tools = {
        "search",
        "access",
        "add_mem",
    }
    graph_tools = {
        "read_mem",
        "edit_mem",
        "add_mem",
    }

    def __init__(self) -> None:
        super().__init__()
        self.last_error: str | None = None

    def _error(self, message: str) -> None:
        self.last_error = message

    def _parse_graph_planner_calls(
        self, tool_name: str, tool_arguments: dict, max_workers_per_planner: int
    ) -> list[ToolRequest]:
        if tool_name == "call_sub":
            subtasks = tool_arguments.get(
                "subtasks", tool_arguments.get("sub_agents", [])
            )
            if not isinstance(subtasks, list):
                self._error("call_sub.subtasks must be a list")
                return []
            if max_workers_per_planner >= 0:
                subtasks = subtasks[:max_workers_per_planner]
            requests = []
            for index, subtask in enumerate(subtasks):
                if not isinstance(subtask, dict):
                    self._error(f"subtasks[{index}] must be an object")
                    continue
                prompt = subtask.get("subtask", subtask.get("prompt", ""))
                if not isinstance(prompt, str) or not prompt.strip():
                    self._error(f"subtasks[{index}].subtask is required")
                    continue
                focus_refs = subtask.get("focus_refs", [])
                if not isinstance(focus_refs, list) or not all(
                    isinstance(ref, str) for ref in focus_refs
                ):
                    self._error(f"subtasks[{index}].focus_refs must be a string list")
                    continue
                requests.append(
                    ToolRequest(
                        name="subtask",
                        arguments={
                            "subtask": prompt.strip(),
                            "focus_refs": focus_refs,
                            "output_contract": subtask.get("output_contract", {}),
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
        if tool_name == "add_mem":
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

        # Validate modes before applying caps, otherwise truncation could hide
        # a mixed-mode call (for example add_mem followed by search).
        if role == "planner":
            modes = {
                "call_sub" if request.name == "subtask" else request.name
                for request in requests
            }
            if len(modes) > 1 or ("edit_mem" in modes and len(requests) != 1):
                self._error("MIXED_TOOL_MODE: planner turn must use one tool mode")
                requests = []
        elif role == "worker":
            modes = {
                "research" if request.name in {"search", "access"} else "add_mem"
                for request in requests
            }
            if len(modes) > 1 or ("add_mem" in modes and len(requests) != 1):
                self._error("MIXED_TOOL_MODE: worker turn must use one tool mode")
                requests = []
        if self.last_error:
            return self.tool_call_regex.sub("", response_text), []

        # Apply per-turn caps after aggregating multiple tags.  A batch mode is
        # still one logical call, while edit_mem/add_mem remain single-call modes.
        if role == "planner" and max_workers_per_planner >= 0:
            if any(request.name == "subtask" for request in requests):
                requests = requests[:max_workers_per_planner]
        elif role == "worker" and any(
            request.name in {"search", "access"} for request in requests
        ):
            requests = requests[:max_toolcall_per_worker]
        content = self.tool_call_regex.sub("", response_text)
        return content, requests
