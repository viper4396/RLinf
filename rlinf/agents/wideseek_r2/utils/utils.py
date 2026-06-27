# Copyright 2025 The RLinf Authors.
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

from typing import Optional

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import ToolRequest
from rlinf.workers.agent.agent_loop import AgentLoopOutput


def build_tool_call_info(role: str, tool_requests: list[ToolRequest]) -> Optional[dict]:
    """Summarize subtask/search/access counts for a set of tool requests.

    Args:
        role: Current role (`planner`, `worker`, or `single`).
        tool_requests: Tool requests parsed from a model response.

    Returns:
        A dict with subtask/search/access counts and the role, or None when there
        are no tool requests.
    """
    if not tool_requests:
        return None

    subtask_count = 0
    search_count = 0
    access_count = 0
    for request in tool_requests:
        if request.name == "subtask":
            subtask_count += 1
        elif request.name == "search":
            search_count += 1
        elif request.name == "access":
            access_count += 1
    return {
        "subtask": subtask_count,
        "search": search_count,
        "access": access_count,
        "role": role,
    }


def set_max_turns(agentloop_config: DictConfig, role: str) -> int:
    """Resolve the maximum number of turns for a role from agent-loop config.

    Args:
        agentloop_config: The ``agentloop`` config section.
        role: Current role (`planner`, `worker`, or `single`).

    Returns:
        The maximum number of turns allowed for the role.
    """
    if role == "planner":
        return agentloop_config.get("max_planner_turns", 10)
    if role == "single":
        return agentloop_config.get("max_sa_turns", 50)
    if role == "worker":
        return agentloop_config.get("max_worker_turns", 20)
    raise ValueError(f"illegal role {role}")


def populate_turn_extra_fields(
    output_buffer: list[AgentLoopOutput],
) -> tuple[int, int]:
    """Populate per-turn extra fields and count valid planner/worker turns.

    For every turn in ``output_buffer`` this records subtask/search/access counts
    and the raw prompt/response text into ``extra_fields``, and counts the turns
    that issued at least one valid tool call per role.

    Args:
        output_buffer: All turns produced for one trajectory.

    Returns:
        Tuple of ``(num_valid_planner_turns, num_valid_worker_turns)``.
    """
    num_valid_planner_turns = 0
    num_valid_worker_turns = 0

    for single_turn_output in output_buffer:
        subtask_count = 0
        search_count = 0
        access_count = 0
        if single_turn_output.tool_call_info is not None:
            role = single_turn_output.tool_call_info.get("role", "")
            subtask_count = single_turn_output.tool_call_info.get("subtask", 0)
            search_count = single_turn_output.tool_call_info.get("search", 0)
            access_count = single_turn_output.tool_call_info.get("access", 0)

            if role == "planner":
                assert subtask_count > 0
                num_valid_planner_turns += 1
            elif role == "worker" or role == "single":
                assert search_count > 0 or access_count > 0
                num_valid_worker_turns += 1
        single_turn_output.extra_fields["subtask_count"] = subtask_count
        single_turn_output.extra_fields["search_count"] = search_count
        single_turn_output.extra_fields["access_count"] = access_count
        single_turn_output.extra_fields["tool_call_info"] = (
            single_turn_output.tool_call_info
        )
        single_turn_output.extra_fields["prompt_text"] = single_turn_output.prompt_text
        single_turn_output.extra_fields["response_text"] = (
            single_turn_output.response_text
        )

    return num_valid_planner_turns, num_valid_worker_turns
