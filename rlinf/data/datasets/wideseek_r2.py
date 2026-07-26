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
import logging
from typing import Union

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from rlinf.data.datasets.item import DatasetItem
from rlinf.data.datasets.reasoning import ReasoningDataset
from rlinf.data.utils import batch_pad_to_fixed_len


def normalize_answer_type(value) -> str:
    """Normalize an answer-structure value used for prompt strategy selection.

    Args:
        value: Expected to name one of the four supported answer structures.

    Returns:
        A lowercase ``item``, ``set``, ``list``, or ``table`` string.

    Raises:
        ValueError: If ``value`` is not a supported answer type.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"item", "set", "list", "table"}:
            return normalized
    raise ValueError(
        f"Unsupported answer_type {value!r}; expected item, set, list, or table."
    )


class WideSeekR2Dataset(ReasoningDataset):
    def __init__(
        self,
        data_paths: Union[str, list[str]],
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(data_paths, config, tokenizer)
        self.default_answer_type = normalize_answer_type(
            config.data.get("answer_type", "table")
        )
        self.unique_columns_key = config.data.get("unique_columns", "unique_columns")
        self.is_gisa = config.data.get("is_gisa", False)

    @staticmethod
    def _record_answer_type(record: dict, default_type: str) -> str:
        """Resolve the search-strategy type for one record.

        An explicit record-level ``answer_type`` has priority. Otherwise the
        dataset-level default is used. Final answers always use Markdown, so
        legacy format metadata does not participate in type selection.
        """
        if "answer_type" in record:
            return normalize_answer_type(record["answer_type"])
        return default_type

    def __getitem__(self, idx):
        """Return a single prompt with its answer payload."""
        record = self.data[idx]
        prompt = record[self.prompt_key]
        answer = record[self.answer_key]

        answer_type = self._record_answer_type(record, self.default_answer_type)
        answer_dict = {
            "answer": answer,
            "unique_columns": record.get(self.unique_columns_key, []),
            "instance_id": record.get("instance_id", idx),
            "answer_type": answer_type,
        }
        if self.is_gisa:
            answer_dict["is_gisa"] = True
        # Try to get evaluation info if available
        evaluation = record.get("evaluation", None)
        if evaluation:
            if isinstance(evaluation, str):
                try:
                    evaluation = json.loads(evaluation)
                except json.JSONDecodeError:
                    pass
        if isinstance(evaluation, dict):
            answer_dict["required"] = evaluation.get("required", [])
        answer = answer_dict

        prompt_tokens, prompt_length = self.encode(prompt)
        prompt_tokens_tensor = torch.as_tensor(prompt_tokens, dtype=torch.int64)

        if prompt_length > self.max_prompt_length:
            logging.warning(
                f"prompt_tokens_tensor length {prompt_length} exceeds the max_prompt_length {self.max_prompt_length}",
            )
            prompt_tokens_tensor = prompt_tokens_tensor[: self.max_prompt_length]
            prompt_length = self.max_prompt_length

        prompt_tokens_tensor = batch_pad_to_fixed_len(
            [prompt_tokens_tensor],
            self.max_prompt_length,
            self.tokenizer.eos_token_id,
            left_pad=True,
        )[0]
        output = DatasetItem(
            prompt=prompt_tokens_tensor,
            length=prompt_length,
            answer=answer,
            idx=idx,
            image_data=[],
        )
        return output
