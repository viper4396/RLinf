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

"""Unit tests for wideseek_r2 answer_mode normalization and <answer> extraction."""

import pytest
from omegaconf import OmegaConf

from rlinf.agents.wideseek_r2.utils.reward import (
    credit_assignment,
    extract_final_answer,
)
from rlinf.data.datasets.wideseek_r2 import WideSeekR2Dataset, normalize_answer_mode


def test_normalize_answer_mode_legacy_bool():
    # Legacy per-record / config boolean is_markdown maps to answer_mode.
    assert normalize_answer_mode(True) == "markdown"
    assert normalize_answer_mode(False) == "boxed"


def test_normalize_answer_mode_strings():
    assert normalize_answer_mode("markdown") == "markdown"
    assert normalize_answer_mode("boxed") == "boxed"
    # Whitespace is trimmed and matching is case-insensitive.
    assert normalize_answer_mode(" Markdown ") == "markdown"
    assert normalize_answer_mode("BOXED") == "boxed"


def test_normalize_answer_mode_invalid_is_rejected():
    for bad in ["table", "", "json", 3, None]:
        with pytest.raises(ValueError):
            normalize_answer_mode(bad)


def test_extract_tag_basic():
    assert extract_final_answer("<answer>hello</answer>", mode="tag") == "hello"


def test_extract_tag_strips_and_handles_think():
    text = "<think>reasoning</think> prelude <answer>\n  the result \n</answer> tail"
    assert extract_final_answer(text, mode="tag") == "the result"


def test_extract_tag_last_match_wins():
    text = "<answer>first</answer> ... <answer>second</answer>"
    assert extract_final_answer(text, mode="tag") == "second"


def test_extract_tag_missing_returns_none():
    assert extract_final_answer("no tags here", mode="tag") is None
    assert extract_final_answer("", mode="tag") is None
    # An unclosed tag is not a valid answer block.
    assert extract_final_answer("<answer>unclosed", mode="tag") is None


def test_credit_assignment_adds_format_bonus_when_format_valid():
    cfg = OmegaConf.create({"format_reward": 0.1})

    assert credit_assignment(cfg, llm_reward=0.8, answer_format=True) == pytest.approx(
        0.9
    )


def test_credit_assignment_preserves_llm_reward_when_format_invalid():
    cfg = OmegaConf.create({"format_reward": 0.1})

    assert credit_assignment(cfg, llm_reward=0.8, answer_format=False) == pytest.approx(
        0.8
    )


def test_config_answer_mode_ignores_config_level_is_markdown():
    # Config-level legacy `is_markdown` must NOT be accepted; default is boxed.
    assert (
        WideSeekR2Dataset._config_answer_mode(OmegaConf.create({"is_markdown": True}))
        == "boxed"
    )
    assert WideSeekR2Dataset._config_answer_mode(OmegaConf.create({})) == "boxed"
    # Only `answer_mode` is honored at config level (even if is_markdown is set).
    assert (
        WideSeekR2Dataset._config_answer_mode(
            OmegaConf.create({"answer_mode": "markdown"})
        )
        == "markdown"
    )
    assert (
        WideSeekR2Dataset._config_answer_mode(
            OmegaConf.create({"answer_mode": "boxed", "is_markdown": True})
        )
        == "boxed"
    )


def test_record_answer_mode_supports_legacy_per_record_is_markdown():
    # Per-record legacy `is_markdown` is still normalized (approved DEC-4 surface).
    assert WideSeekR2Dataset._record_answer_mode({"is_markdown": True}, "boxed") == (
        "markdown"
    )
    assert WideSeekR2Dataset._record_answer_mode(
        {"is_markdown": False}, "markdown"
    ) == ("boxed")
    # Per-record `answer_mode` takes precedence; absent keys fall back to default.
    assert WideSeekR2Dataset._record_answer_mode(
        {"answer_mode": "boxed"}, "markdown"
    ) == ("boxed")
    assert WideSeekR2Dataset._record_answer_mode({}, "markdown") == "markdown"
