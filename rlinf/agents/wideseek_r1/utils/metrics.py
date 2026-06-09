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


def _safe_max(values: list[int]) -> int:
    """Return the max value for a list, or 0 when the list is empty."""
    return max(values) if values else 0


def _add_weighted_mean_metric(
    metrics: dict[str, float],
    key: str,
    numerator: float,
    denominator: float,
) -> None:
    """Record a weighted mean and its backing sum/count statistics.

    Args:
        metrics: Mutable metrics dictionary to update in place.
        key: Visible metric name.
        numerator: Sum of weighted values.
        denominator: Sum of weights.
    """
    metrics[f"__mean__/{key}"] = (float(numerator), float(denominator))


def _compute_tool_call_metrics(
    rollout_batch: dict,
    idx_to_traj: list[int],
    num_trajectories: int,
) -> dict:
    """Compute tool-call metrics for one question group."""
    if num_trajectories <= 0:
        return {}

    turn_subtask_counts = list(rollout_batch.get("turn_subtask_counts", []))
    turn_search_counts = list(rollout_batch.get("turn_search_counts", []))
    turn_access_counts = list(rollout_batch.get("turn_access_counts", []))
    num_valid_planner_turns = int(rollout_batch.get("num_valid_planner_turns", 0))
    num_valid_worker_turns = int(rollout_batch.get("num_valid_worker_turns", 0))

    # idx_to_traj may come from training turns; use common prefix to avoid index mismatch.
    num_mapped_turns = min(
        len(idx_to_traj),
        len(turn_subtask_counts),
        len(turn_search_counts),
        len(turn_access_counts),
    )

    traj_subtask_counts = [0 for _ in range(num_trajectories)]
    traj_search_counts = [0 for _ in range(num_trajectories)]
    traj_access_counts = [0 for _ in range(num_trajectories)]
    for turn_idx in range(num_mapped_turns):
        traj_idx = idx_to_traj[turn_idx]
        if 0 <= traj_idx < num_trajectories:
            traj_subtask_counts[traj_idx] += turn_subtask_counts[turn_idx]
            traj_search_counts[traj_idx] += turn_search_counts[turn_idx]
            traj_access_counts[traj_idx] += turn_access_counts[turn_idx]

    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/traj/mean/subtask",
        sum(traj_subtask_counts),
        num_trajectories,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/traj/mean/search",
        sum(traj_search_counts),
        num_trajectories,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/traj/mean/access",
        sum(traj_access_counts),
        num_trajectories,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/turn/mean/subtask",
        sum(turn_subtask_counts),
        num_valid_planner_turns,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/turn/mean/search",
        sum(turn_search_counts),
        num_valid_worker_turns,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/turn/mean/access",
        sum(turn_access_counts),
        num_valid_worker_turns,
    )

    return metrics


def _compute_mas_turn_metrics(rollout_batch: dict) -> dict:
    """Compute MAS turn metrics for one question group."""
    total_turn_list_metric = rollout_batch.get("total_turn_list_metric")
    if total_turn_list_metric is None:
        return {}

    sum_main_turns = 0
    sum_subagent_turns = 0
    sum_num_subagents = 0
    num_valid_trajs = 0
    for turn_list in total_turn_list_metric:
        if turn_list is None or len(turn_list) == 0:
            continue
        sum_main_turns += turn_list[-1]
        subagent_turns_list = turn_list[:-1]
        sum_subagent_turns += sum(subagent_turns_list)
        sum_num_subagents += len(subagent_turns_list)
        num_valid_trajs += 1

    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_main_agent_turns_per_traj",
        sum_main_turns,
        num_valid_trajs,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_subagent_turns_per_traj",
        sum_subagent_turns,
        num_valid_trajs,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_num_subagents_per_traj",
        sum_num_subagents,
        num_valid_trajs,
    )
    return metrics


def _compute_num_turn_subagents_metrics(rollout_batch: dict) -> dict:
    """Compute per-trajectory num_turn_subagents statistics.

    ``num_turn_subagents`` records how many sub-agents the planner created at
    each turn.  The sum across all planner turns must equal the total number of
    sub-agents (i.e. ``len(total_turn_list) - 1``).
    """
    num_turn_subagents_metric = rollout_batch.get("num_turn_subagents_metric")
    if num_turn_subagents_metric is None:
        return {}

    sum_num_subagents = 0
    sum_num_creation_events = 0
    num_valid_trajs = 0
    for nts in num_turn_subagents_metric:
        if nts is None:
            continue
        s = sum(nts)
        sum_num_subagents += s
        sum_num_creation_events += len(nts)
        num_valid_trajs += 1

    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_num_subagents_per_traj_v2",
        sum_num_subagents,
        num_valid_trajs,
    )
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_subagent_creation_events_per_traj",
        sum_num_creation_events,
        num_valid_trajs,
    )
    return metrics


def _compute_num_effective_subagents_metrics(rollout_batch: dict) -> dict:
    """Compute per-trajectory effective sub-agent statistics.

    An *effective* sub-agent is one that performed at least one ``search``
    tool call during its execution.  ``num_effective_subagents`` has the same
    length as ``num_turn_subagents``.
    """
    num_effective_subagents_metric = rollout_batch.get(
        "num_effective_subagents_metric"
    )
    if num_effective_subagents_metric is None:
        return {}

    sum_effective = 0
    num_valid_trajs = 0
    for nes in num_effective_subagents_metric:
        if nes is None:
            continue
        sum_effective += sum(nes)
        num_valid_trajs += 1

    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_num_effective_subagents_per_traj",
        sum_effective,
        num_valid_trajs,
    )
    return metrics


def _compute_access_search_ratio_metrics(rollout_batch: dict) -> dict:
    """Compute per-trajectory access/search ratio statistics.

    Each element of ``access_search_ratio`` is ``turn_total_access /
    turn_total_search`` across all sub-agents in one planner turn.
    """
    access_search_ratio_metric = rollout_batch.get("access_search_ratio_metric")
    if access_search_ratio_metric is None:
        return {}

    sum_ratio = 0.0
    num_entries = 0
    for asr in access_search_ratio_metric:
        if asr is None:
            continue
        for v in asr:
            sum_ratio += v
            num_entries += 1

    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/avg_access_search_ratio_per_turn",
        sum_ratio,
        num_entries,
    )
    return metrics


def _compute_final_answer_format_metrics(rollout_batch: dict) -> dict:
    """Compute trajectory-level final-answer-format metrics."""
    final_answer_format = rollout_batch.get("final_answer_format")
    if final_answer_format is None:
        return {}

    values = [float(v) for v in final_answer_format]
    metrics: dict[str, float] = {}
    _add_weighted_mean_metric(
        metrics,
        "wideseek_r1/traj/mean/final_answer_format",
        sum(values),
        len(values),
    )
    return metrics


def _compute_reward_metrics(rollout_batch: dict) -> dict:
    """Compute trajectory-level reward metrics."""
    traj_reward_agg = rollout_batch.get("traj_reward_agg_metric")
    outcome_reward = rollout_batch.get("outcome_reward_metric")
    llm_reward = rollout_batch.get("llm_reward_metric")

    metrics: dict[str, float] = {}
    if traj_reward_agg is not None:
        values = [float(v) for v in traj_reward_agg]
        _add_weighted_mean_metric(
            metrics,
            "wideseek_r1/traj/mean/traj_reward_agg",
            sum(values),
            len(values),
        )
    if outcome_reward is not None:
        values = [float(v) for v in outcome_reward]
        _add_weighted_mean_metric(
            metrics,
            "wideseek_r1/traj/mean/outcome_reward",
            sum(values),
            len(values),
        )
    if llm_reward is not None:
        values = [float(v) for v in llm_reward]
        _add_weighted_mean_metric(
            metrics,
            "wideseek_r1/traj/mean/llm_reward",
            sum(values),
            len(values),
        )

    # snapshot F1 scores: compute final-snapshot F1 and snapshot count.
    snapshot_f1_scores = rollout_batch.get("snapshot_f1_scores_metric")
    if snapshot_f1_scores is not None:
        final_f1_values = []
        snapshot_counts = []
        for entry in snapshot_f1_scores:
            if entry is None or not isinstance(entry, list):
                continue
            scores = [e[0] if isinstance(e, (list, tuple)) else float(e)
                      for e in entry]
            final_f1_values.append(scores[-1] if scores else 0.0)
            snapshot_counts.append(len(scores))
        if final_f1_values:
            _add_weighted_mean_metric(
                metrics,
                "wideseek_r1/traj/mean/snapshot_f1_final",
                sum(final_f1_values),
                len(final_f1_values),
            )
        if snapshot_counts:
            _add_weighted_mean_metric(
                metrics,
                "wideseek_r1/traj/mean/snapshot_count",
                sum(snapshot_counts),
                len(snapshot_counts),
            )
    return metrics


def _compute_rollout_metrics(
    rollout_batch: dict,
    idx_to_traj: list[int],
    num_trajectories: int,
) -> dict:
    """Compute all wideseek agent metrics for one question group."""
    metrics = {}
    metrics.update(
        _compute_tool_call_metrics(
            rollout_batch=rollout_batch,
            idx_to_traj=idx_to_traj,
            num_trajectories=num_trajectories,
        )
    )
    metrics.update(_compute_mas_turn_metrics(rollout_batch=rollout_batch))
    metrics.update(_compute_num_turn_subagents_metrics(rollout_batch=rollout_batch))
    metrics.update(_compute_num_effective_subagents_metrics(rollout_batch=rollout_batch))
    metrics.update(_compute_access_search_ratio_metrics(rollout_batch=rollout_batch))
    metrics.update(_compute_final_answer_format_metrics(rollout_batch=rollout_batch))
    metrics.update(_compute_reward_metrics(rollout_batch=rollout_batch))
    return metrics
