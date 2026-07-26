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

import ast
import asyncio
import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

from rlinf.agents.wideseek_r2.utils import reward
from rlinf.agents.wideseek_r2.utils.eval_metrics import aggregate_gisa_metrics
from rlinf.data.datasets.wideseek_r2 import WideSeekR2Dataset


class _Tokenizer:
    eos_token_id = 0
    is_fast = True

    def encode(self, _text):
        return [1, 2]


def _build_dataset(tmp_path, record, *, default_answer_type="table"):
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
                "answer_type": default_answer_type,
                "is_gisa": True,
                "unique_columns": "unique_columns",
            }
        }
    )
    return WideSeekR2Dataset(str(data_path), config, _Tokenizer())


def _canonical(value) -> str:
    text = str(value).strip().lower()
    return (
        text.replace("republic of chile", "chile")
        .replace("santiago de chile", "santiago")
        .replace("plurinational state of bolivia", "bolivia")
    )


class _SemanticJudge:
    def __init__(self):
        self.calls = []

    async def __call__(self, messages):
        self.calls.append(messages)
        user_content = messages[-1]["content"]
        if "idx_0" not in user_content:
            return (
                "Correct"
                if "republic of chile" in user_content.lower()
                and "chile" in user_content.lower()
                else "Incorrect"
            )

        payload_text = user_content[
            user_content.find("{") : user_content.rfind("}") + 1
        ]
        payload = ast.literal_eval(payload_text)
        scores = {
            index: float(_canonical(pair["response"]) == _canonical(pair["target"]))
            for index, pair in payload.items()
        }
        return f"```json\n{json.dumps(scores)}\n```"


def _score_gisa(extract_answer, label_answer, judge):
    return asyncio.run(
        reward.get_final_reward_score(
            origin_question="Name the countries.",
            extract_answer=extract_answer,
            label_answer=label_answer,
            norm_column=False,
            judge_llm_generator=judge,
        )
    )


def test_gisa_record_carries_answer_type(tmp_path):
    record = {
        "question": "Name the colors.",
        "answer": "```markdown\n| Item |\n| --- |\n| red |\n```",
        "unique_columns": ["Item"],
        "answer_type": "set",
    }

    answer = _build_dataset(tmp_path, record)[0].answer

    assert answer["is_gisa"] is True
    assert answer["answer_type"] == "set"


def test_gisa_item_record_uses_unified_markdown_payload(tmp_path):
    record = {
        "question": "Name one color.",
        "answer": "red",
        "unique_columns": None,
        "answer_type": "item",
    }

    answer = _build_dataset(tmp_path, record)[0].answer

    assert answer == {
        "answer": "red",
        "unique_columns": None,
        "instance_id": 0,
        "is_gisa": True,
        "answer_type": "item",
    }


def test_gisa_record_defaults_to_table_type(tmp_path):
    record = {
        "question": "Describe the colors.",
        "answer": "```markdown\n| Item | Value |\n| --- | --- |\n| red | warm |\n```",
    }

    answer = _build_dataset(tmp_path, record)[0].answer

    assert answer["answer_type"] == "table"


def test_gisa_item_uses_semantic_cell_judge():
    judge = _SemanticJudge()
    label_answer = {
        "answer": ["Chile"],
        "is_gisa": True,
        "answer_type": "item",
    }

    result = _score_gisa(
        pd.DataFrame({"Item": ["Republic of Chile"]}),
        label_answer,
        judge,
    )

    assert result == (1.0, True, {"cell_f1": 1.0, "pass": 1.0})
    assert len(judge.calls) == 1


def test_gisa_set_f1_uses_semantic_cell_matching():
    judge = _SemanticJudge()
    label_answer = {
        "answer": ("```markdown\n| Item |\n| --- |\n| Chile |\n| Bolivia |\n```"),
        "is_gisa": True,
        "answer_type": "set",
    }
    prediction = pd.DataFrame(
        {"Item": ["Plurinational State of Bolivia", "Republic of Chile"]}
    )

    score, format_ok, metrics = _score_gisa(prediction, label_answer, judge)

    assert score == 1.0
    assert format_ok is True
    assert metrics == {"cell_f1": 1.0, "pass": 1.0}
    assert len(judge.calls) == 1


def test_gisa_list_preserves_semantic_f1_and_order_score():
    judge = _SemanticJudge()
    label_answer = {
        "answer": ("```markdown\n| Item |\n| --- |\n| Chile |\n| Bolivia |\n```"),
        "is_gisa": True,
        "answer_type": "list",
    }
    reversed_prediction = pd.DataFrame(
        {"Item": ["Plurinational State of Bolivia", "Republic of Chile"]}
    )

    score, format_ok, metrics = _score_gisa(reversed_prediction, label_answer, judge)

    assert score == 1.0
    assert format_ok is True
    assert metrics == {
        "cell_f1": 1.0,
        "order_score": pytest.approx(0.5),
        "pass": 0.0,
    }


def test_gisa_table_uses_cell_and_row_judges():
    judge = _SemanticJudge()
    label_answer = {
        "answer": (
            "```markdown\n| Name | Country |\n| --- | --- |\n"
            "| Alice | Chile |\n| Bob | Bolivia |\n```"
        ),
        "unique_columns": ["Name"],
        "is_gisa": True,
        "answer_type": "table",
    }
    prediction = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Country": ["Republic of Chile", "Peru"],
        }
    )

    score, format_ok, metrics = _score_gisa(prediction, label_answer, judge)

    assert score == pytest.approx(0.75)
    assert format_ok is True
    assert metrics == {
        "cell_f1": pytest.approx(0.75),
        "row_f1": pytest.approx(0.5),
        "pass": 0.0,
    }
    assert len(judge.calls) == 3


def test_gisa_requires_judge_model():
    label_answer = {
        "answer": ["Chile"],
        "is_gisa": True,
        "answer_type": "item",
    }

    assert _score_gisa(pd.DataFrame({"Item": ["Chile"]}), label_answer, None) == (
        0.0,
        False,
        {},
    )


@pytest.mark.parametrize(
    ("answer_type", "invalid_answer"),
    [
        ("item", pd.DataFrame({"Item": ["Chile", "Argentina"]})),
        ("set", pd.DataFrame({"Item": ["Chile"], "Other": ["extra"]})),
        ("list", pd.DataFrame({"Answer": ["Chile"]})),
        ("table", pd.DataFrame()),
    ],
)
def test_gisa_rejects_invalid_markdown_shape_before_judge(answer_type, invalid_answer):
    async def judge(_messages):
        raise AssertionError("Invalid GISA output must not call the judge")

    label_answer = {
        "answer": ["Chile"],
        "is_gisa": True,
        "answer_type": answer_type,
    }

    assert _score_gisa(invalid_answer, label_answer, judge) == (0.0, False, {})


def test_item_markdown_output_parses_to_one_cell_table():
    answer = reward.extract_final_answer(
        "```markdown\n| Item |\n| :--- |\n| Chile |\n```"
    )

    assert isinstance(answer, pd.DataFrame)
    assert answer.to_dict(orient="records") == [{"Item": "Chile"}]


def test_non_gisa_item_still_uses_judge_model():
    calls = []

    async def judge(messages):
        calls.append(messages)
        return "Correct"

    result = asyncio.run(
        reward.get_final_reward_score(
            origin_question="Name one country.",
            extract_answer=pd.DataFrame({"Item": ["Chile"]}),
            label_answer={"answer": ["Chile"], "answer_type": "item"},
            norm_column=False,
            judge_llm_generator=judge,
        )
    )

    assert result == (1.0, True, {})
    assert len(calls) == 1


def test_gisa_metrics_preserve_f1_order_and_pass():
    raw_results = [
        {
            "answer": {"answer_type": "table"},
            "samples": [
                {
                    "gisa_metrics": {
                        "cell_f1": 1.0,
                        "row_f1": 1.0,
                        "pass": 1.0,
                    }
                },
                {
                    "gisa_metrics": {
                        "cell_f1": 0.75,
                        "row_f1": 0.5,
                        "pass": 0.0,
                    }
                },
            ],
        },
        {
            "answer": {"answer_type": "list"},
            "samples": [
                {
                    "gisa_metrics": {
                        "cell_f1": 1.0,
                        "order_score": 0.5,
                        "pass": 0.0,
                    }
                },
                {
                    "gisa_metrics": {
                        "cell_f1": 1.0,
                        "order_score": 1.0,
                        "pass": 1.0,
                    }
                },
            ],
        },
        {
            "answer": {"answer_type": "item"},
            "samples": [
                {"gisa_metrics": {"cell_f1": 1.0, "pass": 1.0}},
                {"gisa_metrics": {"cell_f1": 0.0, "pass": 0.0}},
            ],
        },
        {
            "answer": {"answer_type": "set"},
            "samples": [
                {"gisa_metrics": {"cell_f1": 0.5, "pass": 0.0}},
                {"gisa_metrics": {"cell_f1": 1.0, "pass": 1.0}},
            ],
        },
    ]

    metrics = aggregate_gisa_metrics(raw_results, enabled=True)

    assert metrics == {
        "item_f1@1": pytest.approx(0.875),
        "avg_item_f1@k": pytest.approx(0.78125),
        "max_item_f1@k": 1.0,
        "pass@1": 0.5,
        "avg@k": 0.5,
        "pass@k": 1.0,
        "row_f1@1": 1.0,
        "avg_row_f1@k": 0.75,
        "max_row_f1@k": 1.0,
        "order_score@1": 0.5,
        "avg_order_score@k": 0.75,
        "max_order_score@k": 1.0,
    }


def test_non_gisa_results_do_not_include_gisa_metrics():
    assert aggregate_gisa_metrics([], enabled=False) == {}
