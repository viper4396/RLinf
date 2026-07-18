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

import pytest

from rlinf.agents.wideseek_r2.utils.prompt_utils import (
    get_prompt_planner,
    get_prompt_single_agent,
)


@pytest.mark.parametrize(
    ("task_type", "call_strategy"),
    [
        ("Item", "after results return, launch next-hop or verification calls"),
        ("Set", "launch one sub-agent per bucket"),
        ("List", "First call agents for ranking sources or candidate segments"),
        ("Table", "First call agents to establish the row universe"),
    ],
)
def test_planner_noshot_includes_task_type_strategies(
    task_type: str, call_strategy: str
):
    messages = get_prompt_planner(
        question="Research this task.",
        answer_mode="markdown",
        add_few_shot=False,
        max_workers_per_planner=-1,
    )

    system_prompt = messages[0]["content"]
    assert "# Task-Type Strategies" in system_prompt
    assert f"**{task_type}**" in system_prompt
    assert call_strategy in system_prompt
    assert "controls sub-agent calls and verification" in system_prompt
    assert "GISA" not in system_prompt
    assert "There is NO limit on the number of sub-agents" in system_prompt
    assert "```markdown" in system_prompt


@pytest.mark.parametrize(
    ("task_type", "search_strategy"),
    [
        ("Item", "first unresolved hop"),
        ("Set", "authoritative definition or enumeration first"),
        ("List", "before collecting items"),
        ("Table", "complete row universe first"),
    ],
)
def test_single_agent_noshot_includes_task_type_strategies(
    task_type: str, search_strategy: str
):
    messages = get_prompt_single_agent(
        question="Research this task.",
        answer_mode="boxed",
        add_few_shot=False,
    )

    system_prompt = messages[0]["content"]
    assert "# Task-Type Strategies" in system_prompt
    assert f"**{task_type}**" in system_prompt
    assert search_strategy in system_prompt
    assert "controls search order and verification" in system_prompt
    assert "GISA" not in system_prompt
    assert "\\boxed" in system_prompt


def test_task_type_strategy_guidance_is_limited_to_noshot_prompts():
    planner_system = get_prompt_planner(
        question="Research this task.",
        answer_mode="boxed",
        add_few_shot=True,
        max_workers_per_planner=4,
    )[0]["content"]
    single_system = get_prompt_single_agent(
        question="Research this task.",
        answer_mode="markdown",
        add_few_shot=True,
    )[0]["content"]

    assert "# Task-Type Strategies" not in planner_system
    assert "# Task-Type Strategies" not in single_system
