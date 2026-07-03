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


def normalize_answer_mode(value) -> str:
    """Normalize a config/record answer-mode value to ``markdown`` or ``boxed``.

    Accepts the new string form (``"markdown"`` / ``"boxed"``) as well as the
    legacy boolean ``is_markdown`` form (``True`` -> ``"markdown"``,
    ``False`` -> ``"boxed"``) so that existing datasets keep working.

    Args:
        value: A string answer mode or a legacy boolean ``is_markdown`` flag.

    Returns:
        Either ``"markdown"`` or ``"boxed"``.

    Raises:
        ValueError: If the value cannot be interpreted as a supported mode.
    """
    if isinstance(value, bool):
        return "markdown" if value else "boxed"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("markdown", "boxed"):
            return normalized
    raise ValueError(
        f"Unsupported answer_mode {value!r}; expected 'markdown' or 'boxed' "
        "(or the legacy boolean is_markdown)."
    )


class WideSeekR2Dataset(ReasoningDataset):
    def __init__(
        self,
        data_paths: Union[str, list[str]],
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(data_paths, config, tokenizer)
        # Config-level answer mode is resolved ONLY from `data.answer_mode`
        # (default "boxed"); a config-level legacy `is_markdown` key is
        # intentionally NOT accepted. Per-record legacy `is_markdown` is still
        # normalized in `_record_answer_mode` (the approved DEC-4 compatibility
        # surface for existing hybrid data).
        self.answer_mode = self._config_answer_mode(config.data)
        self.unique_columns_key = config.data.get("unique_columns", "unique_columns")
        self.is_hybrid = config.data.get("is_hybrid", False)

    @staticmethod
    def _config_answer_mode(data_cfg) -> str:
        """Resolve the dataset-level default answer mode from config.

        Only ``data.answer_mode`` is honored (defaulting to ``"boxed"`` when
        unset); a config-level legacy ``is_markdown`` key is NOT accepted.
        """
        return normalize_answer_mode(data_cfg.get("answer_mode", "boxed"))

    @staticmethod
    def _record_answer_mode(record: dict, default: str) -> str:
        """Resolve the answer mode for one hybrid record.

        Prefers a per-record ``answer_mode``, then a legacy per-record
        ``is_markdown`` flag, and finally the dataset-level ``default``.
        """
        if "answer_mode" in record:
            return normalize_answer_mode(record["answer_mode"])
        if "is_markdown" in record:
            return normalize_answer_mode(record["is_markdown"])
        return default

    def __getitem__(self, idx):
        """Return a single prompt with its answer payload."""
        record = self.data[idx]
        prompt = record[self.prompt_key]
        answer = record[self.answer_key]

        answer_mode = (
            self._record_answer_mode(record, self.answer_mode)
            if self.is_hybrid
            else self.answer_mode
        )

        if answer_mode == "markdown":
            answer_dict = {
                "answer": answer,
                "unique_columns": record.get(self.unique_columns_key, []),
                "answer_mode": answer_mode,
                "instance_id": record.get("instance_id", idx),
            }
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
        else:
            answer = {
                "answer": answer if isinstance(answer, list) else [answer],
                "answer_mode": answer_mode,
                "instance_id": record.get("instance_id", idx),
            }

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
