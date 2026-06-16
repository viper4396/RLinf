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

import math
from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",  # "trajectory" or "turn"
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx
    - Trajectories 0..(group_size-1) belong to question 0, etc.

    Two advantage computation modes (set via ``advantage_mode`` in config):

    1. **"trajectory"** — discounted trajectory-level GRPO.
       ``traj_reward_i = Σ(k=0..T_i-2) γ^(T_i-k) × r_i,k + outcome_i``
       z-score across G trajectories; all turns in the same traj share A_i.

    2. **"turn"** — turn-level GRPO with discounted cumulative + outcome blend.
       Non-outcome turns: z-score across all non-outcome turns in the question.
       Outcome rewards: z-score across G trajectories.
       ``D_i,t = Σ(k=t..T_i-1) γ^(T_i-k) × R_i,k / √(T_i - t)``
       ``A_i,t = ω × D_i,t + (1-ω) × R_o_i``  (t < T_i-1)
       ``A_i,t = R_o_i``                        (t = T_i-1)

    Both modes require ``planner_turn_idx`` mapping (turn → planner turn index
    within its trajectory).

    Args:
        rewards: Shape [num_sequence, 1] (num_sequence = total turns).
        loss_mask: Shape [seq_len, num_sequence].
        group_size: Number of trajectories per question.
        idx_to_traj: List mapping turn_idx → global_traj_idx.
        advantage_mode: ``"trajectory"`` or ``"turn"``.

    Keyword Args:
        planner_turn_idx: List mapping turn_idx → planner turn index (required).
        gamma: Discount factor (default 0.9).
        omega: Outcome blending weight (default 0.5).

    Returns:
        advantages: Shape [seq_len, num_sequence].
    """
    num_sequence = len(idx_to_traj)

    rewards_flat = rewards.squeeze(-1)

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )

    planner_turn_idx = kwargs.get("planner_turn_idx", None)
    gamma = kwargs.get("gamma", 0.9)
    omega = kwargs.get("omega", 0.5)

    assert planner_turn_idx is not None, (
        "grpo_dynamic requires planner_turn_idx mapping"
    )

    turn_to_question = torch.tensor(
        [idx_to_traj[i] // group_size for i in range(num_sequence)],
        dtype=torch.long,
        device=rewards.device,
    )

    for question_idx in range(num_questions):
        question_mask = turn_to_question == question_idx
        q_indices = question_mask.nonzero(as_tuple=True)[0]

        # Build per-trajectory info for this question.
        traj_turns: dict[int, list[tuple[int, int, float]]] = {}
        for seq_idx in q_indices.tolist():
            traj = idx_to_traj[seq_idx]
            pti = planner_turn_idx[seq_idx]
            if traj not in traj_turns:
                traj_turns[traj] = []
            traj_turns[traj].append((seq_idx, pti, float(rewards_flat[seq_idx])))

        # Separate outcome (last planner turn) from non-outcome.
        outcome_entries: list[tuple[int, int, float]] = []
        non_outcome_entries: list[tuple[int, float]] = []
        for traj, entries in traj_turns.items():
            entries.sort(key=lambda x: x[1])
            for j, (seq_idx, pti, r) in enumerate(entries):
                if j == len(entries) - 1:
                    outcome_entries.append((traj, seq_idx, r))
                else:
                    non_outcome_entries.append((seq_idx, r))

        if advantage_mode == "trajectory":
            # ---- trajectory-level GRPO ----
            # When reward_mode="trajectory", all turns in a traj share the same
            # traj_reward_agg, so use the reward directly without discount-sum.
            # When reward_mode="turn", per-turn rewards differ and need to be
            # aggregated via discounted sum.
            reward_mode = kwargs.get("reward_mode", "turn")
            if reward_mode == "trajectory":
                trajectory_rewards = torch.zeros(
                    num_trajectories, dtype=rewards.dtype, device=rewards.device
                )
                for traj, entries in traj_turns.items():
                    entries.sort(key=lambda x: x[1])
                    trajectory_rewards[traj] = entries[-1][2]
            else:
                # traj_reward_i = Σ(k=0..T_i-2) γ^(T_i - k) × r_i,k + outcome_i
                trajectory_rewards = torch.zeros(
                    num_trajectories, dtype=rewards.dtype, device=rewards.device
                )
                for traj, entries in traj_turns.items():
                    entries.sort(key=lambda x: x[1])
                    T_i = len(entries)
                    total = 0.0
                    for j, (seq_idx, pti, r) in enumerate(entries):
                        if j == T_i - 1:
                            total += r  # outcome
                        else:
                            total += (gamma ** (T_i - j)) * r
                    trajectory_rewards[traj] = total

            grouped = trajectory_rewards.view(num_questions, group_size)
            mean = grouped.mean(dim=-1, keepdim=True)
            std = grouped.std(dim=-1, keepdim=True)
            normalized = ((grouped - mean) / (std + 1e-6)).view(-1)
            for turn_idx in range(num_sequence):
                if turn_to_question[turn_idx] == question_idx:
                    turn_advantages[turn_idx] = normalized[idx_to_traj[turn_idx]]

        elif advantage_mode == "turn":
            # ---- turn-level z-score + discounted cumulative + outcome blend ----
            # Non-outcome: z-score across all non-outcome turns in the question.
            if non_outcome_entries:
                non_rewards = torch.tensor(
                    [r for _, r in non_outcome_entries],
                    dtype=rewards.dtype,
                    device=rewards.device,
                )
                non_mean = non_rewards.mean()
                non_std = non_rewards.std()
                for seq_idx, r in non_outcome_entries:
                    turn_advantages[seq_idx] = (r - non_mean) / (non_std + 1e-6)

            # Outcome: z-score across G trajectories.
            R_o = torch.zeros(
                num_trajectories, dtype=rewards.dtype, device=rewards.device
            )
            if outcome_entries:
                out_rewards = torch.tensor(
                    [r for _, _, r in outcome_entries],
                    dtype=rewards.dtype,
                    device=rewards.device,
                )
                o_mean = out_rewards.mean()
                o_std = out_rewards.std()
                for traj, seq_idx, r in outcome_entries:
                    R_o[traj] = (r - o_mean) / (o_std + 1e-6)

            # D_i,t = Σ(k=t..T_i-1) γ^(T_i - k) × R_i,k / √(T_i - t)
            # A_i,t = ω × D_i,t + (1-ω) × R_o_i  ...  A_i,T_i-1 = R_o_i
            for traj, entries in traj_turns.items():
                entries.sort(key=lambda x: x[1])
                T_i = len(entries)

                R_list = [
                    float(R_o[traj])
                    if j == T_i - 1
                    else float(turn_advantages[seq_idx])
                    for j, (seq_idx, pti, r) in enumerate(entries)
                ]

                for t in range(T_i):
                    D = 0.0
                    tn = T_i - t
                    for k in range(t, T_i):
                        D += (gamma ** (T_i - k)) * R_list[k]
                    D /= math.sqrt(tn)
                    seq_idx = entries[t][0]
                    if t == T_i - 1:
                        turn_advantages[seq_idx] = R_o[traj]
                    else:
                        turn_advantages[seq_idx] = omega * D + (1.0 - omega) * R_o[traj]

        else:
            raise ValueError(
                f"Invalid advantage_mode: {advantage_mode}. "
                "Must be 'trajectory' or 'turn'"
            )

    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("raw")
def compute_raw_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
):
    """
    Return raw rewards or normalized rewards.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        normalize_advantages (bool): Whether to normalize advantages.

    Returns:
        torch.Tensor: advantages
    """
    if rewards.ndim == 2:
        rewards = rewards.reshape(-1)
    advantages = rewards.unsqueeze(0).expand_as(loss_mask) * loss_mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid = advantages[loss_mask.bool()]
        if valid.numel() > 0:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)

    return advantages, None
