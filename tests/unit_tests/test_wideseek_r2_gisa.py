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

import asyncio
import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

from rlinf.agents.wideseek_r2.utils import reward
from rlinf.data.datasets.wideseek_r2 import WideSeekR2Dataset


class _Tokenizer:
    eos_token_id = 0
    is_fast = True

    def encode(self, _text):
        return [1, 2]


def _build_dataset(tmp_path, record, *, answer_mode="boxed"):
    data_path = tmp_path / "gisa.jsonl"
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    config = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 8,
                "prompt_key": "question",
                "answer_key": "answer",
                "apply_chat_template": False,
                "filter_prompt_by_length": False,
                "data_size": -1,
                "answer_mode": answer_mode,
                "is_hybrid": True,
                "is_gisa": True,
                "unique_columns": "unique_columns",
            }
        }
    )
    return WideSeekR2Dataset(str(data_path), config, _Tokenizer())


def test_gisa_markdown_record_carries_answer_type(tmp_path):
    record = {
        "question": "Name the colors.",
        "answer": "```markdown\n| Item |\n| --- |\n| red |\n```",
        "unique_columns": ["Item"],
        "is_markdown": True,
        "answer_type": "set",
    }

    answer = _build_dataset(tmp_path, record)[0].answer

    assert answer["answer_mode"] == "markdown"
    assert answer["is_gisa"] is True
    assert answer["answer_type"] == "set"


def test_gisa_boxed_record_carries_item_type(tmp_path):
    record = {
        "question": "Name one color.",
        "answer": "red",
        "unique_columns": None,
        "is_markdown": False,
    }

    answer = _build_dataset(tmp_path, record, answer_mode="markdown")[0].answer

    assert answer == {
        "answer": ["red"],
        "answer_mode": "boxed",
        "instance_id": 0,
        "is_gisa": True,
        "answer_type": "item",
    }


def test_gisa_markdown_record_defaults_to_table_type(tmp_path):
    record = {
        "question": "Describe the colors.",
        "answer": "```markdown\n| Item | Value |\n| --- | --- |\n| red | warm |\n```",
        "is_markdown": True,
    }

    answer = _build_dataset(tmp_path, record)[0].answer

    assert answer["answer_mode"] == "markdown"
    assert answer["answer_type"] == "table"


def test_gisa_markdown_record_requires_supported_answer_type(tmp_path):
    record = {
        "question": "Name one color.",
        "answer": "```markdown\n| Item |\n| --- |\n| red |\n```",
        "is_markdown": True,
        "answer_type": "item",
    }

    dataset = _build_dataset(tmp_path, record)

    with pytest.raises(ValueError, match="table, set, or list"):
        dataset[0]


def _gisa_markdown_label(answer_type):
    return {
        "answer": (
            "```markdown\n| Item |\n| :--- |\n| Chile |\n| Argentina |\n"
            "| Bolivia |\n| Ecuador |\n```"
        ),
        "unique_columns": ["Item"],
        "answer_mode": "markdown",
        "is_gisa": True,
        "answer_type": answer_type,
    }


def _score_gisa(extract_answer, label_answer, answer_mode="markdown"):
    return asyncio.run(
        reward.get_final_reward_score(
            origin_question="Name the countries.",
            extract_answer=extract_answer,
            label_answer=label_answer,
            answer_mode=answer_mode,
            norm_column=False,
            judge_llm_generator=None,
        )
    )


def test_gisa_set_em_ignores_header_order_and_duplicates():
    label_answer = _gisa_markdown_label("set")
    predicted_answer = pd.DataFrame(
        {
            "Arbitrary header": [
                "Ecuador",
                "Chile",
                "Bolivia",
                "Argentina",
                "Chile",
            ]
        }
    )

    assert _score_gisa(predicted_answer, label_answer) == (1.0, True)

    predicted_answer.loc[len(predicted_answer)] = "Peru"
    assert _score_gisa(predicted_answer, label_answer) == (0.0, True)


def test_gisa_list_em_ignores_header_but_requires_order():
    label_answer = _gisa_markdown_label("list")
    ordered_answer = pd.DataFrame(
        {
            "Arbitrary header": [
                "Chile",
                "Argentina",
                "Bolivia",
                "Ecuador",
            ]
        }
    )
    reordered_answer = ordered_answer.iloc[::-1].reset_index(drop=True)

    assert _score_gisa(ordered_answer, label_answer) == (1.0, True)
    assert _score_gisa(reordered_answer, label_answer) == (0.0, True)


def test_gisa_table_em_compares_headers_rows_and_cells():
    label_answer = {
        "answer": (
            "```markdown\n| Name | Country |\n| --- | --- |\n"
            "| Alice | Chile |\n| Bob | Bolivia |\n```"
        ),
        "unique_columns": ["Name"],
        "answer_mode": "markdown",
        "is_gisa": True,
        "answer_type": "table",
    }
    exact_answer = pd.DataFrame(
        {"Name": ["Alice", "Bob"], "Country": ["Chile", "Bolivia"]}
    )
    changed_header = exact_answer.rename(columns={"Name": "Person"})
    reordered_answer = exact_answer.iloc[::-1].reset_index(drop=True)

    assert _score_gisa(exact_answer, label_answer) == (1.0, True)
    assert _score_gisa(changed_header, label_answer) == (0.0, True)
    assert _score_gisa(reordered_answer, label_answer) == (0.0, True)


def test_gisa_item_uses_em_without_judge_model():
    label_answer = {
        "answer": ["Chile"],
        "answer_mode": "boxed",
        "is_gisa": True,
        "answer_type": "item",
    }

    assert _score_gisa("Chile", label_answer, answer_mode="boxed") == (1.0, True)
    assert _score_gisa("chile", label_answer, answer_mode="boxed") == (0.0, True)


def test_non_gisa_item_still_uses_judge_model():
    calls = []

    async def judge(messages):
        calls.append(messages)
        return "Correct"

    score, format_ok = asyncio.run(
        reward.get_final_reward_score(
            origin_question="Name one country.",
            extract_answer="Chile",
            label_answer={"answer": ["Chile"], "answer_mode": "boxed"},
            answer_mode="boxed",
            norm_column=False,
            judge_llm_generator=judge,
        )
    )

    assert (score, format_ok) == (1.0, True)
    assert len(calls) == 1


def test_gisa_em_does_not_call_configured_judge_model():
    async def judge(_messages):
        raise AssertionError("GISA answer EM must not call the judge model")

    score, format_ok = asyncio.run(
        reward.get_final_reward_score(
            origin_question="Name one country.",
            extract_answer="Chile",
            label_answer={
                "answer": ["Chile"],
                "answer_mode": "boxed",
                "is_gisa": True,
                "answer_type": "item",
            },
            answer_mode="boxed",
            norm_column=False,
            judge_llm_generator=judge,
        )
    )

    assert (score, format_ok) == (1.0, True)
