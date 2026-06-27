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

from rlinf.agents.wideseek_r2.utils.reward import extract_final_answer
from rlinf.data.datasets.wideseek_r2 import normalize_answer_mode


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
