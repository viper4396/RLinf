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

import asyncio
from types import SimpleNamespace

import torch

from rlinf.agents.wideseek_r2.utils.reward import (
    compute_planner_hindsight_weights,
    evaluate_worker_quality,
)
from rlinf.algorithms.advantages import compute_gigpo_advantages


def test_hierarchical_gigpo_assigns_planner_turn_and_worker_agent_advantages():
    # Two trajectories for one question. Trajectory 0 has two turns for worker
    # 1, verifying that a worker receives one shared agent-level advantage.
    idx_to_traj = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    role_ids = [0, 1, 1, 1, 0, 0, 1, 1, 0]
    idx_to_sub_traj = [0, 1, 1, 2, 0, 0, 1, 2, 0]
    planner_turn_idx = [0, -1, -1, -1, 1, 0, -1, -1, 1]
    parent_planner_turn_idx = [-1, 0, 0, 0, -1, -1, 0, 0, -1]
    planner_hindsight_weight = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    worker_quality_score = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    worker_quality_valid = [False, True, True, True, False, False, True, True, False]
    worker_format_valid = [False] * len(idx_to_traj)
    rewards = torch.tensor([[1.0] if traj == 0 else [0.0] for traj in idx_to_traj])
    loss_mask = torch.ones(3, len(idx_to_traj))

    advantages, _ = compute_gigpo_advantages(
        rewards=rewards,
        loss_mask=loss_mask,
        group_size=2,
        idx_to_traj=idx_to_traj,
        role_ids=role_ids,
        idx_to_sub_traj=idx_to_sub_traj,
        planner_turn_idx=planner_turn_idx,
        parent_planner_turn_idx=parent_planner_turn_idx,
        planner_hindsight_weight=planner_hindsight_weight,
        worker_quality_score=worker_quality_score,
        worker_quality_valid=worker_quality_valid,
        worker_format_valid=worker_format_valid,
        planner_hindsight_gamma=1.0,
        worker_parent_adv_weight=0.5,
        worker_local_adv_weight=0.5,
        worker_format_reward=0.0,
        worker_quality_baseline=0.5,
        worker_quality_scale=0.5,
    )

    trajectory_advantage = 0.5 / (torch.sqrt(torch.tensor(0.5)) + 1e-6)
    planner_0 = trajectory_advantage * (2.0 / 1.5)
    planner_1 = trajectory_advantage * (1.0 / 1.5)
    low_quality_worker = 0.5 * planner_0 - 0.5
    high_quality_worker = 0.5 * planner_0 + 0.5

    assert torch.allclose(advantages[:, 0], torch.full((3,), planner_0), atol=1e-5)
    assert torch.allclose(advantages[:, 4], torch.full((3,), planner_1), atol=1e-5)
    assert torch.allclose(
        advantages[:, 1], torch.full((3,), low_quality_worker), atol=1e-5
    )
    assert torch.equal(advantages[:, 1], advantages[:, 2])
    assert torch.allclose(
        advantages[:, 3], torch.full((3,), high_quality_worker), atol=1e-5
    )
    # The losing trajectory keeps negative planner credit on both turns.
    assert torch.all(advantages[:, 5] < 0)
    assert torch.all(advantages[:, 8] < 0)


def test_worker_quality_judge_returns_weighted_grounded_score():
    async def judge(_messages):
        return """```json
        {"relevance": 1, "groundedness": 0.5, "coverage": 0.8, "usefulness": 0.75}
        ```"""

    score, valid = asyncio.run(
        evaluate_worker_quality(
            main_question="question",
            subtask="subtask",
            worker_summary="supported summary",
            evidence_context="supporting evidence",
            judge_llm_generator=judge,
        )
    )

    assert valid is True
    assert abs(score - 0.75) < 1e-6


def test_planner_hindsight_keeps_final_turn_weight_one():
    turns = [
        SimpleNamespace(prompt_text="state 0", response_ids=[1, 2]),
        SimpleNamespace(prompt_text="state 1", response_ids=[3]),
        SimpleNamespace(prompt_text="final", response_ids=[4]),
    ]

    async def compute_logprobs(_prompt, action_tokens, _temperature):
        value = -0.1 if action_tokens[0] == 1 else -1.0
        return [value] * len(action_tokens)

    weights = asyncio.run(
        compute_planner_hindsight_weights(
            planner_turns=turns,
            final_answer="answer",
            outcome_reward=1.0,
            temperature=1.0,
            c_min=0.1,
            c_max=5.0,
            compute_logprobs=compute_logprobs,
        )
    )

    assert len(weights) == 3
    assert weights[0] > weights[1]
    assert abs((weights[0] + weights[1]) / 2 - 1.0) < 1e-6
    assert weights[2] == 1.0
