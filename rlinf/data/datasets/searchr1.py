# Copyright 2026 The RLinf Authors.
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

"""Dataset prompt adapter for Search-R1 checkpoints."""

from typing import Any

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from rlinf.data.datasets.reasoning import ReasoningDataset

SEARCHR1_LOCAL_RAG_PROMPT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
<|im_end|>
<|im_start|>assistant
<think>"""


def format_searchr1_prompt(
    question: str,
    *,
    answer_type: str | None = None,
    is_markdown: bool = False,
) -> str:
    """Wrap a raw question in the prompt used to train Search-R1."""
    if is_markdown:
        structure = str(answer_type or "table").strip().lower()
        schema_instruction = (
            " Use exactly one column named Item."
            if structure in {"set", "list"}
            else " Use the exact column names requested by the question."
        )
        format_instruction = (
            "\nReturn the final answer inside <answer> and </answer> as exactly "
            f"one fenced Markdown table representing a {structure} answer. "
            f"Preserve the requested row order for list answers.{schema_instruction}"
        )
        question = f"{question.strip()}{format_instruction}"
    return SEARCHR1_LOCAL_RAG_PROMPT.format(question=question.strip())


class SearchR1Dataset(ReasoningDataset):
    """Reasoning dataset that wraps raw questions with Search-R1 ChatML."""

    def __init__(
        self,
        data_paths: str | list[str],
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        self.is_gisa = bool(config.data.get("is_gisa", False))
        self.unique_columns_key = str(
            config.data.get("unique_columns", "unique_columns")
        )
        if config.data.get("apply_chat_template", False):
            raise ValueError(
                "SearchR1Dataset supplies its own ChatML; "
                "data.apply_chat_template must be false"
            )
        super().__init__(data_paths=data_paths, config=config, tokenizer=tokenizer)

    def _load_data(self):
        records = super()._load_data()
        for record in records:
            question = record[self.prompt_key]
            if not isinstance(question, str):
                raise ValueError("Search-R1 questions must be strings")
            if "<|im_start|>" not in question:
                if self.is_gisa:
                    record["_searchr1_raw_question"] = question.strip()
                record[self.prompt_key] = format_searchr1_prompt(
                    question,
                    answer_type=record.get("answer_type"),
                    is_markdown=bool(record.get("is_markdown", False))
                    if self.is_gisa
                    else False,
                )
        return records

    def __getitem__(self, idx: int):
        """Return a prompt and preserve GISA structural scoring metadata."""
        item = super().__getitem__(idx)
        if not self.is_gisa:
            return item

        record: dict[str, Any] = self.data[idx]
        answer_type = str(record.get("answer_type", "table")).strip().lower()
        is_markdown = bool(record.get("is_markdown", answer_type != "item"))
        supported_types = {"table", "set", "list"} if is_markdown else {"item"}
        if answer_type not in supported_types:
            expected = "table, set, or list" if is_markdown else "item"
            raise ValueError(
                f"GISA {'markdown' if is_markdown else 'plain'} records require "
                f"answer_type to be one of {expected}; got {answer_type!r} "
                f"at index {idx}."
            )

        item.answer = {
            "answer": record[self.answer_key],
            "answer_type": answer_type,
            "is_markdown": is_markdown,
            "is_gisa": True,
            "instance_id": record.get("instance_id", record.get("id", idx)),
            "unique_columns": record.get(self.unique_columns_key, []),
            "question_type": record.get("question_type"),
            "topic": record.get("topic"),
            "question": record.get("_searchr1_raw_question"),
        }
        return item
