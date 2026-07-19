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

STRATEGY_HEADERS = {
    "item": "# Answer-Type Strategy: Item",
    "set": "# Answer-Type Strategy: Set",
    "list": "# Answer-Type Strategy: List",
    "table": "# Answer-Type Strategy: Table",
}


def _assert_only_strategy(system_prompt: str, answer_type: str) -> None:
    assert STRATEGY_HEADERS[answer_type] in system_prompt
    for other_type, header in STRATEGY_HEADERS.items():
        if other_type != answer_type:
            assert header not in system_prompt
    assert "GISA" not in system_prompt


@pytest.mark.parametrize(
    ("answer_type", "call_strategy"),
    [
        ("item", "Do not delegate later hops whose inputs are still unknown"),
        ("set", "create one sub-agent per bucket"),
        ("list", "cover independent rank ranges, pages, or candidate segments"),
        ("table", "call sub-agents to discover and verify the complete row"),
    ],
)
def test_planner_noshot_injects_selected_answer_type_strategy(
    answer_type: str, call_strategy: str
):
    messages = get_prompt_planner(
        question="Research this task.",
        answer_mode="markdown",
        add_few_shot=False,
        max_workers_per_planner=-1,
        answer_type=answer_type,
    )

    system_prompt = messages[0]["content"]
    _assert_only_strategy(system_prompt, answer_type)
    assert call_strategy in system_prompt
    assert "There is NO limit on the number of sub-agents" in system_prompt
    assert "```markdown" in system_prompt


@pytest.mark.parametrize(
    ("answer_type", "search_strategy"),
    [
        ("item", "Search for that prerequisite"),
        ("set", "search each bucket systematically"),
        ("list", "follow its pages or sections in order"),
        ("table", "First find and verify the complete row/entity universe"),
    ],
)
def test_single_agent_noshot_injects_selected_answer_type_strategy(
    answer_type: str, search_strategy: str
):
    messages = get_prompt_single_agent(
        question="Research this task.",
        answer_mode="boxed",
        add_few_shot=False,
        answer_type=answer_type,
    )

    system_prompt = messages[0]["content"]
    _assert_only_strategy(system_prompt, answer_type)
    assert search_strategy in system_prompt
    assert "\\boxed" in system_prompt


def test_noshot_prompt_answer_type_falls_back_from_answer_mode():
    planner_system = get_prompt_planner(
        question="Research this task.",
        answer_mode="boxed",
        add_few_shot=False,
        max_workers_per_planner=4,
    )[0]["content"]
    single_system = get_prompt_single_agent(
        question="Research this task.",
        answer_mode="markdown",
        add_few_shot=False,
    )[0]["content"]

    _assert_only_strategy(planner_system, "item")
    _assert_only_strategy(single_system, "table")


def test_task_type_strategy_guidance_is_limited_to_noshot_prompts():
    planner_system = get_prompt_planner(
        question="Research this task.",
        answer_mode="boxed",
        add_few_shot=True,
        max_workers_per_planner=4,
        answer_type="set",
    )[0]["content"]
    single_system = get_prompt_single_agent(
        question="Research this task.",
        answer_mode="markdown",
        add_few_shot=True,
        answer_type="list",
    )[0]["content"]

    assert "# Answer-Type Strategy:" not in planner_system
    assert "# Answer-Type Strategy:" not in single_system


def test_noshot_prompt_rejects_unsupported_answer_type():
    with pytest.raises(ValueError, match="Unsupported answer_type"):
        get_prompt_single_agent(
            question="Research this task.",
            answer_mode="markdown",
            add_few_shot=False,
            answer_type="graph",
        )
