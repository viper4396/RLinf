# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from rlinf.agents.wideseek_r1.utils.prompt_utils import (
    get_prompt_planner,
    get_prompt_single_agent,
)


@pytest.mark.parametrize("builder", [get_prompt_planner, get_prompt_single_agent])
@pytest.mark.parametrize(
    ("answer_type", "order_instruction"),
    [
        ("set", "Row order does not matter"),
        ("list", "in the exact required order"),
    ],
)
def test_gisa_collection_prompt_requires_one_column_pipe_table(
    builder, answer_type, order_instruction
):
    messages = builder(
        "Research this task.",
        is_markdown=True,
        language="en",
        add_few_shot=False,
        answer_type=answer_type,
        is_gisa=True,
    )

    system_prompt = messages[0]["content"]
    assert "# Few-shot Examples" not in system_prompt
    assert "<answer>" in system_prompt
    assert "output only one fenced Markdown pipe table" in system_prompt
    assert "| Item |" in system_prompt
    assert order_instruction in system_prompt
    assert "pipe table, NOT JSON" in system_prompt


def test_gisa_table_keeps_existing_markdown_format():
    system_prompt = get_prompt_planner(
        "Research this task.",
        is_markdown=True,
        language="en",
        add_few_shot=False,
        answer_type="table",
        is_gisa=True,
    )[0]["content"]

    assert "```markdown\n{data_content}\n```" in system_prompt
    assert "| Item |" not in system_prompt


def test_non_gisa_prompt_keeps_legacy_format_and_few_shot_selection():
    markdown_system = get_prompt_planner(
        "Research this task.",
        is_markdown=True,
        language="en",
        answer_type="set",
    )[0]["content"]
    boxed_system = get_prompt_planner(
        "Research this task.",
        is_markdown=False,
        language="en",
    )[0]["content"]

    assert "# Few-shot Examples" in markdown_system
    assert "| Item |" not in markdown_system
    assert "# Few-shot Examples" not in boxed_system
    assert "\\boxed" in boxed_system


def test_gisa_chinese_list_prompt_uses_strict_order_instruction():
    system_prompt = get_prompt_single_agent(
        "调研这个问题。",
        is_markdown=True,
        language="zh",
        add_few_shot=False,
        answer_type="list",
        is_gisa=True,
    )[0]["content"]

    assert "严格保持题目要求的顺序" in system_prompt
    assert "| Item |" in system_prompt
