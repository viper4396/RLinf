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

import json

import pytest
from omegaconf import OmegaConf

from rlinf.data.datasets.wideseek_r1 import WideSeekR1Dataset


class _Tokenizer:
    eos_token_id = 0
    is_fast = True

    def encode(self, _text):
        return [1, 2]


def _build_dataset(tmp_path, record, *, is_markdown=True, is_hybrid=False):
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
                "is_markdown": is_markdown,
                "is_hybrid": is_hybrid,
                "is_gisa": True,
                "unique_columns": "unique_columns",
            }
        }
    )
    return WideSeekR1Dataset(str(data_path), config, _Tokenizer())


def test_gisa_markdown_record_carries_prompt_metadata(tmp_path):
    dataset = _build_dataset(
        tmp_path,
        {
            "question": "Name the countries.",
            "answer": "```markdown\n| Item |\n| --- |\n| Chile |\n```",
            "unique_columns": ["Item"],
            "answer_type": "SET",
        },
    )

    assert dataset[0].answer == {
        "answer": "```markdown\n| Item |\n| --- |\n| Chile |\n```",
        "unique_columns": ["Item"],
        "is_markdown": True,
        "instance_id": 0,
        "language": "en",
        "answer_type": "set",
        "is_gisa": True,
    }


def test_gisa_markdown_record_defaults_to_table(tmp_path):
    dataset = _build_dataset(
        tmp_path,
        {"question": "Describe the country.", "answer": "reference"},
    )

    assert dataset[0].answer["answer_type"] == "table"


def test_gisa_markdown_rejects_item_answer_type(tmp_path):
    dataset = _build_dataset(
        tmp_path,
        {
            "question": "Name one country.",
            "answer": "reference",
            "answer_type": "item",
        },
    )

    with pytest.raises(ValueError, match="table, set, or list"):
        dataset[0]


def test_gisa_hybrid_boxed_record_carries_item_type(tmp_path):
    dataset = _build_dataset(
        tmp_path,
        {
            "question": "Name one country.",
            "answer": "Chile",
            "is_markdown": False,
            "answer_type": "item",
        },
        is_hybrid=True,
    )

    assert dataset[0].answer == {
        "answer": ["Chile"],
        "is_markdown": False,
        "instance_id": 0,
        "language": "en",
        "answer_type": "item",
        "is_gisa": True,
    }
