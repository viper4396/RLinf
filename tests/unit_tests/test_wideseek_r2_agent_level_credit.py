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

import torch

from rlinf.agents.wideseek_r2.utils.reward import evaluate_worker_quality
from rlinf.algorithms.advantages import compute_gigpo_advantages


def test_gigpo_assigns_one_advantage_per_planner_and_worker_agent():
    # Two trajectories for one question. Trajectory 0 has two turns for worker
    # 1, verifying that a worker receives one shared agent-level advantage.
    idx_to_traj = [0, 0, 0, 0, 0, 1, 1, 1, 1]
    role_ids = [0, 1, 1, 1, 0, 0, 1, 1, 0]
    idx_to_sub_traj = [0, 1, 1, 2, 0, 0, 1, 2, 0]
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
        worker_quality_score=worker_quality_score,
        worker_quality_valid=worker_quality_valid,
        worker_format_valid=worker_format_valid,
        worker_local_weight=0.5,
        planner_worker_quality_weight=0.25,
        worker_format_reward=0.0,
    )

    trajectory_advantage = 0.5 / (torch.sqrt(torch.tensor(0.5)) + 1e-6)
    low_quality_worker = trajectory_advantage - 0.5
    high_quality_worker = trajectory_advantage + 0.5

    assert torch.allclose(
        advantages[:, 0], torch.full((3,), trajectory_advantage), atol=1e-5
    )
    assert torch.equal(advantages[:, 0], advantages[:, 4])
    assert torch.allclose(
        advantages[:, 1], torch.full((3,), low_quality_worker), atol=1e-5
    )
    assert torch.equal(advantages[:, 1], advantages[:, 2])
    assert torch.allclose(
        advantages[:, 3], torch.full((3,), high_quality_worker), atol=1e-5
    )
    assert torch.equal(advantages[:, 5], advantages[:, 8])
    assert torch.allclose(
        advantages[:, 5], torch.full((3,), -trajectory_advantage), atol=1e-5
    )


def test_agent_credit_uses_format_and_neutral_invalid_quality():
    idx_to_traj = [0, 0, 1, 1]
    role_ids = [0, 1, 0, 1]
    idx_to_sub_traj = [0, 1, 0, 1]
    rewards = torch.zeros(4, 1)
    loss_mask = torch.ones(2, 4)

    advantages, _ = compute_gigpo_advantages(
        rewards=rewards,
        loss_mask=loss_mask,
        group_size=2,
        idx_to_traj=idx_to_traj,
        role_ids=role_ids,
        idx_to_sub_traj=idx_to_sub_traj,
        worker_quality_score=[0.0, 0.0, 0.0, 0.0],
        worker_quality_valid=[False, False, False, False],
        worker_format_valid=[False, True, False, False],
        worker_local_weight=0.5,
        planner_worker_quality_weight=0.25,
        worker_format_reward=0.1,
    )

    assert torch.allclose(advantages[:, 0], torch.full((2,), 0.025))
    assert torch.allclose(advantages[:, 1], torch.full((2,), 0.05))
    assert torch.equal(advantages[:, 2], torch.zeros(2))
    assert torch.equal(advantages[:, 3], torch.zeros(2))


def test_planner_without_workers_and_single_agent_use_trajectory_advantage():
    advantages, _ = compute_gigpo_advantages(
        rewards=torch.tensor([[1.0], [0.0]]),
        loss_mask=torch.ones(2, 2),
        group_size=2,
        idx_to_traj=[0, 1],
        role_ids=[0, 2],
        idx_to_sub_traj=[0, 0],
        worker_quality_score=[0.0, 0.0],
        worker_quality_valid=[False, False],
        worker_format_valid=[False, False],
        worker_local_weight=0.5,
        planner_worker_quality_weight=0.25,
        worker_format_reward=0.1,
    )

    trajectory_advantage = 0.5 / (torch.sqrt(torch.tensor(0.5)) + 1e-6)
    assert torch.allclose(
        advantages[:, 0], torch.full((2,), trajectory_advantage), atol=1e-5
    )
    assert torch.allclose(
        advantages[:, 1], torch.full((2,), -trajectory_advantage), atol=1e-5
    )


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
