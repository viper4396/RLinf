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

"""Unit tests for WideSeek-R2 eval metric aggregation."""

import pytest
from omegaconf import OmegaConf

from rlinf.agents.wideseek_r2.eval_runner import WideSeekR2AgentEvalRunner


def _build_runner(answer_mode: str, raw_results: list[dict]) -> WideSeekR2AgentEvalRunner:
    runner = WideSeekR2AgentEvalRunner.__new__(WideSeekR2AgentEvalRunner)
    runner.cfg = OmegaConf.create({"data": {"answer_mode": answer_mode}})
    runner.accumulated_raw_results = raw_results
    return runner


def test_markdown_eval_aggregates_f1_metrics_only():
    runner = _build_runner(
        "markdown",
        [
            {
                "group_size": 2,
                "answer": {"answer_mode": "markdown"},
                "samples": [
                    {"llm_reward": 0.2, "final_answer_format": 1, "turns": []},
                    {"llm_reward": 0.6, "final_answer_format": 1, "turns": []},
                ],
            },
            {
                "group_size": 1,
                "answer": {"answer_mode": "markdown"},
                "samples": [
                    {"llm_reward": 1.0, "final_answer_format": 1, "turns": []},
                ],
            },
        ],
    )

    _, metrics = runner._aggregate_all_results()

    assert metrics["item_f1@1"] == pytest.approx(0.6)
    assert metrics["avg_item_f1@k"] == pytest.approx(0.7)
    assert metrics["max_item_f1@k"] == pytest.approx(0.8)
    assert "pass@1" not in metrics
    assert "avg@k" not in metrics
    assert "pass@k" not in metrics


def test_boxed_eval_aggregates_pass_metrics_only():
    runner = _build_runner(
        "boxed",
        [
            {
                "group_size": 2,
                "answer": {"answer_mode": "boxed"},
                "samples": [
                    {"llm_reward": 0.0, "final_answer_format": 0, "turns": []},
                    {"llm_reward": 1.0, "final_answer_format": 1, "turns": []},
                ],
            },
            {
                "group_size": 1,
                "answer": {"answer_mode": "boxed"},
                "samples": [
                    {"llm_reward": 0.2, "final_answer_format": 1, "turns": []},
                ],
            },
        ],
    )

    _, metrics = runner._aggregate_all_results()

    assert metrics["pass@1"] == pytest.approx(0.5)
    assert metrics["avg@k"] == pytest.approx(0.35)
    assert metrics["pass@k"] == pytest.approx(1.0)
    assert "item_f1@1" not in metrics
    assert "avg_item_f1@k" not in metrics
    assert "max_item_f1@k" not in metrics
