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
    answer_mode: str,
    is_hybrid: bool = False,
):
    data_path = tmp_path / "data.jsonl"
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
                "is_hybrid": is_hybrid,
            }
        }
    )
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
        answer_mode="boxed",
    )

    assert answer["answer_type"] == answer_type


@pytest.mark.parametrize(
    ("record_metadata", "config_mode", "expected_type"),
    [
        ({"answer_mode": "boxed"}, "markdown", "item"),
        ({"answer_mode": "markdown"}, "boxed", "table"),
        ({"answer_mode": "boxed", "is_markdown": True}, "markdown", "item"),
        ({"is_markdown": False}, "markdown", "item"),
        ({"is_markdown": True}, "boxed", "table"),
        ({}, "boxed", "item"),
        ({}, "markdown", "table"),
    ],
)
def test_missing_answer_type_falls_back_from_format_metadata(
    tmp_path, record_metadata, config_mode, expected_type
):
    answer = _load_answer(
        tmp_path,
        {
            "question": "Research this task.",
            "answer": "reference",
            **record_metadata,
        },
        answer_mode=config_mode,
    )

    assert answer["answer_type"] == expected_type


def test_explicit_answer_type_has_priority_over_format_metadata(tmp_path):
    answer = _load_answer(
        tmp_path,
        {
            "question": "Research this task.",
            "answer": "reference",
            "answer_type": "list",
            "answer_mode": "boxed",
            "is_markdown": False,
        },
        answer_mode="boxed",
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
            answer_mode="markdown",
        )
