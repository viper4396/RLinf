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

from typing import Any

from omegaconf import DictConfig

from rlinf.algorithms.searchr1_scoring import compute_score
from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.scheduler import Channel, Worker


def assign_searchr1_rewards(
    rollout_result: DynamicRolloutResult,
    answer: Any,
    *,
    reward_scale: float = 1.0,
    expose_reference: bool = False,
) -> list[float]:
    """Score trajectories and place each final reward on its terminal turn.

    Ground truth is provided only to this reward-side function. It is never
    added to a policy, teacher-planner, or reward-model input.

    Args:
        rollout_result: One question's grouped multi-turn rollout.
        answer: Ground-truth answer or accepted-answer list.
        reward_scale: Multiplicative scale for exact-match rewards.
        expose_reference: Whether evaluation output should retain the answer.

    Returns:
        One final score for each trajectory in the rollout group.
    """
    extra_fields_traj = rollout_result.extra_fields_traj or {}
    response_texts = extra_fields_traj.get("response_text")
    if response_texts is None or len(response_texts) != rollout_result.group_size:
        raise ValueError("Search-R1 reward requires one response_text per trajectory")

    final_scores = [
        float(compute_score(text, answer, do_print=False)) * reward_scale
        for text in response_texts
    ]
    turn_ids = rollout_result.extra_fields_train.get(
        "planner_turn_idx", list(range(rollout_result.num_sequence))
    )
    terminal_flags = rollout_result.extra_fields_train.get(
        "is_terminal", [False] * rollout_result.num_sequence
    )
    if len(turn_ids) != rollout_result.num_sequence:
        raise ValueError("Search-R1 turn IDs must align with retained sequences")
    if len(terminal_flags) != rollout_result.num_sequence:
        raise ValueError("Search-R1 terminal flags must align with retained sequences")
    rewards = [0.0] * rollout_result.num_sequence

    for trajectory_idx, final_score in enumerate(final_scores):
        sequence_indices = [
            sequence_idx
            for sequence_idx, owner in enumerate(rollout_result.idx_to_traj)
            if owner == trajectory_idx
        ]
        terminal_indices = [
            sequence_idx
            for sequence_idx in sequence_indices
            if terminal_flags[sequence_idx]
        ]
        candidates = terminal_indices or sequence_indices
        if not candidates:
            continue
        terminal_idx = max(candidates, key=lambda idx: turn_ids[idx])
        rewards[terminal_idx] = final_score
        rollout_result.is_end[terminal_idx] = True

    rollout_result.rewards = rewards
    extra_fields_traj["llm_reward"] = final_scores
    rollout_result.extra_fields_traj = extra_fields_traj
    rollout_result.extra_fields_group = rollout_result.extra_fields_group or {}
    if expose_reference:
        rollout_result.extra_fields_group["answer"] = answer
    else:
        rollout_result.extra_fields_group.pop("answer", None)
    return final_scores


class SearchR1RewardWorker(Worker):
    """Dynamic exact-match reward worker with an isolated reference channel."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        reward_cfg = cfg.get("reward", {})
        self.reward_scale = float(reward_cfg.get("reward_scale", 1.0))
        self.expose_reference = cfg.runner.task_type == "reasoning_eval"

    def init_worker(self) -> None:
        """Initialize the stateless rule-based reward worker."""

    @Worker.timer("compute_rewards")
    def compute_rewards(
        self,
        input_channel: Channel,
        output_channel: Channel,
        reference_channel: Channel,
        total_batch_size: int | None = None,
    ) -> None:
        """Join opaque rollout IDs with GT references and compute rewards."""
        if total_batch_size is None:
            total_batch_size = int(self.cfg.data.rollout_batch_size)

        references: dict[str, Any] = {}
        while len(references) < total_batch_size:
            reference_batch = reference_channel.get()
            reference_ids = reference_batch["reference_ids"]
            answers = reference_batch["answers"]
            if len(reference_ids) != len(answers):
                raise ValueError("Search-R1 reference IDs and answers must align")
            for reference_id, answer in zip(reference_ids, answers):
                if reference_id in references:
                    raise ValueError(
                        f"Duplicate Search-R1 reference ID: {reference_id}"
                    )
                references[reference_id] = answer

        consumed_ids: set[str] = set()
        while len(consumed_ids) < total_batch_size:
            rollout_result: DynamicRolloutResult = input_channel.get()
            extra_fields_group = rollout_result.extra_fields_group or {}
            reference_id = extra_fields_group.get("reference_id")
            if reference_id not in references:
                raise ValueError(
                    f"Unknown Search-R1 rollout reference ID: {reference_id}"
                )
            if reference_id in consumed_ids:
                raise ValueError(f"Duplicate Search-R1 rollout ID: {reference_id}")

            assign_searchr1_rewards(
                rollout_result,
                references[reference_id],
                reward_scale=self.reward_scale,
                expose_reference=self.expose_reference,
            )
            consumed_ids.add(reference_id)
            output_channel.put(rollout_result, async_op=True)
