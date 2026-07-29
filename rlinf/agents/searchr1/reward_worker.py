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

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from omegaconf import DictConfig

from rlinf.agents.searchr1.judge import JudgeRecord, SearchR1JudgeClient
from rlinf.agents.searchr1.teacher_planner import extract_searchr1_question
from rlinf.agents.wideseek_r1.utils.reward import extract_final_answer
from rlinf.agents.wideseek_r2.utils.reward import (
    evaluate_gisa_with_llm_judge,
    verify_answer_with_llm_judge,
)
from rlinf.algorithms.searchr1_scoring import compute_score, extract_solution
from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.scheduler import Channel, Worker


async def score_searchr1_gisa_response(
    response_text: str,
    answer: dict[str, Any],
    question: str,
    judge_llm_generator: Callable[[list[dict[str, str]]], Awaitable[str]],
) -> tuple[float, float, bool, dict[str, float]]:
    """Score one Search-R1 response using the GISA semantic LLM judge."""
    answer_payload = extract_solution(response_text)
    if answer_payload is None:
        return 0.0, 0.0, False, {}

    answer_type = str(answer.get("answer_type", "table"))
    if answer_type == "item":
        score = await verify_answer_with_llm_judge(
            question=question,
            predicted_answer=answer_payload.strip(),
            correct_answer=answer.get("answer"),
            judge_llm_generator=judge_llm_generator,
            answer_type="item",
        )
        metrics = {"cell_f1": float(score), "pass": float(score)}
        return float(score), float(score), True, metrics

    prediction = extract_final_answer(
        answer_payload,
        mode="markdown",
        strict=False,
    )
    score, format_ok, metrics = await evaluate_gisa_with_llm_judge(
        question=question,
        extract_answer=prediction,
        label_answer=answer,
        judge_llm_generator=judge_llm_generator,
    )
    pass_score = float(metrics.get("pass", 0.0))
    return float(score), pass_score, format_ok, metrics


def assign_searchr1_rewards(
    rollout_result: DynamicRolloutResult,
    answer: Any,
    *,
    reward_scale: float = 1.0,
    expose_reference: bool = False,
    judge_scores: list[float | None] | None = None,
    judge_responses: list[str] | None = None,
    gisa_scores: list[float] | None = None,
    gisa_metrics: list[dict[str, float]] | None = None,
    gisa_format_ok: list[bool] | None = None,
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

    is_gisa = isinstance(answer, dict) and bool(answer.get("is_gisa", False))
    if is_gisa:
        if judge_scores is not None:
            raise ValueError("Search-R1 GISA scoring cannot use the binary judge")
        if (
            gisa_scores is None
            or gisa_metrics is None
            or gisa_format_ok is None
            or len(gisa_scores) != rollout_result.group_size
            or len(gisa_metrics) != rollout_result.group_size
            or len(gisa_format_ok) != rollout_result.group_size
        ):
            raise ValueError(
                "Search-R1 GISA scores, metrics, and format flags must align "
                "with the trajectory group"
            )
        final_scores = [float(score) * reward_scale for score in gisa_scores]
        em_scores = [
            float(metrics.get("pass", 0.0)) * reward_scale for metrics in gisa_metrics
        ]
    else:
        em_scores = [
            float(compute_score(text, answer, do_print=False)) * reward_scale
            for text in response_texts
        ]
        if judge_scores is not None and len(judge_scores) != rollout_result.group_size:
            raise ValueError("Search-R1 judge scores must align with trajectory group")
        final_scores = [
            em_score if judge_score is None else float(judge_score) * reward_scale
            for em_score, judge_score in zip(
                em_scores,
                judge_scores or [None] * rollout_result.group_size,
            )
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
    extra_fields_traj["R_final"] = final_scores
    extra_fields_traj["em_reward"] = em_scores
    if gisa_metrics is not None:
        extra_fields_traj["gisa_metrics"] = gisa_metrics
        extra_fields_traj["gisa_format_ok"] = gisa_format_ok
    if judge_scores is not None:
        extra_fields_traj["judge_reward"] = judge_scores
        extra_fields_traj["judge_response"] = judge_responses
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
        self.is_gisa = bool(cfg.data.get("is_gisa", False))
        judge_cfg = reward_cfg.get("judge", {})
        self.use_judge = bool(judge_cfg.get("enabled", False))
        if self.is_gisa and not self.use_judge:
            raise ValueError(
                "Search-R1 GISA semantic scoring requires reward.judge.enabled=true"
            )
        self.judge_client = SearchR1JudgeClient(cfg) if self.use_judge else None

    def init_worker(self) -> None:
        """Initialize the stateless rule-based reward worker."""

    async def _score_gisa_pending(
        self,
        pending: list[tuple[DynamicRolloutResult, dict[str, Any]]],
    ) -> list[tuple[float, float, bool, dict[str, float]]]:
        """Score one reward batch with a shared judge session and semaphore."""
        if self.judge_client is None:
            raise ValueError("Search-R1 GISA requires an initialized judge client")

        async with self.judge_client.generator() as judge_generate:
            coroutines = []
            for rollout_result, answer in pending:
                extra_fields_traj = rollout_result.extra_fields_traj or {}
                response_texts = extra_fields_traj.get("response_text", [])
                prompt_texts = extra_fields_traj.get(
                    "prompt_text", [""] * rollout_result.group_size
                )
                question = str(answer.get("question") or "")
                for prompt_text, response_text in zip(prompt_texts, response_texts):
                    coroutines.append(
                        score_searchr1_gisa_response(
                            response_text,
                            answer,
                            question or extract_searchr1_question(prompt_text),
                            judge_generate,
                        )
                    )
            return await asyncio.gather(*coroutines)

    @Worker.timer("compute_rewards")
    async def compute_rewards(
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
        pending: list[tuple[DynamicRolloutResult, Any]] = []
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

            consumed_ids.add(reference_id)
            if self.judge_client is None:
                assign_searchr1_rewards(
                    rollout_result,
                    references[reference_id],
                    reward_scale=self.reward_scale,
                    expose_reference=self.expose_reference,
                )
                output_channel.put(rollout_result, async_op=True)
                continue
            pending.append((rollout_result, references[reference_id]))

        judge_outputs: list[tuple[float | None, str]] = []
        gisa_outputs: list[tuple[float, float, bool, dict[str, float]]] = []
        if self.is_gisa:
            gisa_outputs = await self._score_gisa_pending(pending)
        elif self.judge_client is not None:
            judge_records = []
            for rollout_result, answer in pending:
                extra_fields_traj = rollout_result.extra_fields_traj or {}
                response_texts = extra_fields_traj.get("response_text", [])
                prompt_texts = extra_fields_traj.get(
                    "prompt_text", [""] * rollout_result.group_size
                )
                for prompt_text, response_text in zip(prompt_texts, response_texts):
                    judge_records.append(
                        JudgeRecord(
                            question=extract_searchr1_question(prompt_text),
                            predicted_answer=extract_solution(response_text),
                            correct_answer=(
                                answer[0]
                                if isinstance(answer, list) and len(answer) == 1
                                else answer
                            ),
                        )
                    )
            judge_outputs = await self.judge_client.score_many(judge_records)

        judge_offset = 0
        gisa_offset = 0
        for rollout_result, answer in pending:
            group_judgements = (
                judge_outputs[judge_offset : judge_offset + rollout_result.group_size]
                if judge_outputs
                else []
            )
            judge_offset += rollout_result.group_size
            group_gisa = (
                gisa_outputs[gisa_offset : gisa_offset + rollout_result.group_size]
                if gisa_outputs
                else []
            )
            gisa_offset += rollout_result.group_size
            assign_searchr1_rewards(
                rollout_result,
                answer,
                reward_scale=self.reward_scale,
                expose_reference=self.expose_reference,
                judge_scores=(
                    [score for score, _ in group_judgements]
                    if group_judgements
                    else None
                ),
                judge_responses=(
                    [response for _, response in group_judgements]
                    if group_judgements
                    else None
                ),
                gisa_scores=(
                    [score for score, _, _, _ in group_gisa] if group_gisa else None
                ),
                gisa_metrics=(
                    [metrics for _, _, _, metrics in group_gisa] if group_gisa else None
                ),
                gisa_format_ok=(
                    [format_ok for _, _, format_ok, _ in group_gisa]
                    if group_gisa
                    else None
                ),
            )
            output_channel.put(rollout_result, async_op=True)
