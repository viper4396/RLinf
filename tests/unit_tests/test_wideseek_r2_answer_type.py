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

import json

import pytest
from omegaconf import OmegaConf

from rlinf.data.datasets.wideseek_r2 import WideSeekR2Dataset


class _Tokenizer:
    eos_token_id = 0
    is_fast = True

    def encode(self, _text):
        return [1, 2]


def _load_answer(
    tmp_path,
    record: dict,
    *,
    default_answer_type: str | None = None,
):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    data_config = {
        "max_prompt_length": 8,
        "prompt_key": "question",
        "answer_key": "answer",
        "apply_chat_template": False,
        "filter_prompt_by_length": False,
        "data_size": -1,
    }
    if default_answer_type is not None:
        data_config["answer_type"] = default_answer_type
    config = OmegaConf.create({"data": data_config})
    dataset = WideSeekR2Dataset(str(data_path), config, _Tokenizer())
    return dataset[0].answer


@pytest.mark.parametrize("answer_type", ["item", "set", "list", "table"])
def test_explicit_answer_type_is_normalized_and_preserved(tmp_path, answer_type):
    answer = _load_answer(
        tmp_path,
        {
            "question": "Research this task.",
            "answer": "reference",
            "answer_type": answer_type.upper(),
        },
    )

    assert answer["answer_type"] == answer_type
    assert "answer_mode" not in answer


@pytest.mark.parametrize("default_answer_type", ["item", "set", "list", "table"])
def test_missing_answer_type_uses_dataset_default(tmp_path, default_answer_type):
    answer = _load_answer(
        tmp_path,
        {
            "question": "Research this task.",
            "answer": "reference",
        },
        default_answer_type=default_answer_type,
    )

    assert answer["answer_type"] == default_answer_type


def test_missing_record_and_dataset_answer_type_defaults_to_table(tmp_path):
    answer = _load_answer(
        tmp_path,
        {
            "question": "Research this task.",
            "answer": "reference",
        },
    )

    assert answer["answer_type"] == "table"


def test_legacy_format_metadata_does_not_affect_answer_type(tmp_path):
    answer = _load_answer(
        tmp_path,
        {
            "question": "Research this task.",
            "answer": "reference",
            "answer_mode": "boxed",
            "is_markdown": False,
        },
        default_answer_type="table",
    )

    assert answer["answer_type"] == "table"
    assert "answer_mode" not in answer
    assert "is_markdown" not in answer


def test_explicit_answer_type_has_priority_over_dataset_default(tmp_path):
    answer = _load_answer(
        tmp_path,
        {
            "question": "Research this task.",
            "answer": "reference",
            "answer_type": "list",
        },
        default_answer_type="item",
    )

    assert answer["answer_type"] == "list"


def test_invalid_explicit_answer_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported answer_type"):
        _load_answer(
            tmp_path,
            {
                "question": "Research this task.",
                "answer": "reference",
                "answer_type": "graph",
            },
        )
