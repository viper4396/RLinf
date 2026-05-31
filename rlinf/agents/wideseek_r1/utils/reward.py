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
import copy
import json
import re
from io import StringIO
from typing import Awaitable, Callable

import pandas as pd
from omegaconf import DictConfig

from rlinf.workers.agent.agent_loop import AgentLoopOutput


def credit_assignment(
    agentloop_config: DictConfig,
    output_buffer: list[AgentLoopOutput],
    llm_reward,
    succ_end,
    answer_format,
    num_turn_subagents: list[int] | None = None,
    num_effective_subagents: list[int] | None = None,
    access_search_ratio: list[float] | None = None,
    llm_turn_rewards: list[float] | None = None,
    hind_weights: list[float] | None = None,
):
    """Assign turn-level or trajectory-level rewards.

    Two modes, controlled by ``reward_mode`` config:

    **"turn"** (default)
      Per-planner-turn rewards using either LLM-judge scores or formula-based
      parallelism + completion.  The last planner turn gets the outcome reward.

    **"trajectory"**
      Single trajectory-level reward computed from aggregates, then broadcast
      to all turns.  ``traj_reward = llm_reward + parallelism_bonus +
      completion_bonus - length_penalty``.

    When ``hind_weights`` are provided, turn rewards are scaled by
    ``rho_i,t = clip(hind_i,t / mean(hind), c_min, c_max)``.

    Args:
        agentloop_config: Agent-loop config.
        output_buffer: All turns of one trajectory.
        llm_reward: Answer-level reward from ``get_final_reward_score``.
        succ_end: Whether the agent produced a valid answer tag.
        answer_format: Whether final-answer extraction succeeded.
        num_turn_subagents: Per-planner-turn sub-agent counts.
        num_effective_subagents: Per-planner-turn effective sub-agent counts.
        access_search_ratio: Per-planner-turn access / search ratios.
        llm_turn_rewards: Optional LLM-judge per-turn scores.
        hind_weights: Optional hindsight importance weights (rho_i,t).

    Returns:
        ``(output_buffer, train_buffer, final_answer_format, turn_rewards)``.
    """
    reward_mode = agentloop_config.get("reward_mode", "turn")
    length_penalty = 0.0

    length_limit = agentloop_config.get("length_limit", 5000)
    max_length_limit = agentloop_config.get("max_length_limit", 7000)
    length_p = agentloop_config.get("length_penalty", 0.0)

    max_subagents = agentloop_config.get("max_workers_per_planner", 10)
    parallelism_weight = agentloop_config.get("parallelism_weight", 0.0)
    completion_weight = agentloop_config.get("completion_weight", 0.0)
    max_turn_scale = agentloop_config.get("max_turn_scale", 1.0)
    max_planner_turns = agentloop_config.get("max_planner_turns", 10)
    epsilon = agentloop_config.get("parallelism_epsilon", 0.01)
    c_min = agentloop_config.get("hind_clip_min", 0.1)
    c_max = agentloop_config.get("hind_clip_max", 5.0)

    # --- length penalty ------------------------------------------------------
    max_response_len = max(
        (len(t.response_ids) for t in output_buffer if t.response_ids), default=0
    )
    if max_response_len > length_limit:
        t_frac = (max_response_len - length_limit) / (max_length_limit - length_limit)
        t_frac = max(0.0, min(1.0, t_frac))
        length_penalty = t_frac * length_p

    P = len(num_turn_subagents) if num_turn_subagents else 0

    # trajectory-level aggregate bonuses (used by both modes)
    p_bonus, c_bonus = _compute_traj_bonuses(
        num_turn_subagents, num_effective_subagents,
        output_buffer, max_subagents, epsilon,
    )
    traj_reward_agg = (
        (llm_reward + parallelism_weight * p_bonus + completion_weight * c_bonus - length_penalty)
        if (succ_end and answer_format)
        else 0.0
    )

    if reward_mode == "trajectory":
        use_hind = hind_weights is not None and len(hind_weights) >= P - 1
        use_llm_rm = (
            llm_turn_rewards is not None and len(llm_turn_rewards) >= P - 1
        )

        if use_hind or use_llm_rm:
            outcome_reward = traj_reward_agg
            gamma = agentloop_config.get("gamma", 0.9)

            planner_turn_rewards: list[float] = []
            for i in range(P):
                if i == P - 1:
                    planner_turn_rewards.append(outcome_reward)
                elif use_hind:
                    rho = max(c_min, min(c_max, hind_weights[i]))
                    tn = P - i - 1
                    reward = rho * (gamma ** tn) * traj_reward_agg
                    planner_turn_rewards.append(reward)
                elif use_llm_rm:
                    reward = float(llm_turn_rewards[i])
                    planner_turn_rewards.append(reward)

            # Scale all turn rewards when total planner turns exceeds soft limit.
            if P > max_planner_turns:
                planner_turn_rewards = [r * max_turn_scale for r in planner_turn_rewards]

            turn_rewards = []
            planner_idx = -1
            for turn in output_buffer:
                role = turn.extra_fields.get("role", "")
                if role in ("planner", "single"):
                    planner_idx += 1
                if 0 <= planner_idx < P:
                    turn_rewards.append(planner_turn_rewards[planner_idx])
                else:
                    turn_rewards.append(0.0)
        else:
            turn_rewards = [traj_reward_agg] * len(output_buffer)

    else:
        # ---- turn-level reward (existing logic) -----------------------------
        use_llm_rm = (
            llm_turn_rewards is not None and len(llm_turn_rewards) >= P - 1
        )
        if use_llm_rm:
            outcome_reward = traj_reward_agg
        else:
            outcome_reward = (
                (llm_reward - length_penalty) if (succ_end and answer_format) else 0.0
            )

        planner_turn_rewards: list[float] = []
        for i in range(P):
            if i == P - 1:
                planner_turn_rewards.append(outcome_reward)
                continue

            if use_llm_rm:
                reward = float(llm_turn_rewards[i])
            else:
                n_total = num_turn_subagents[i] if num_turn_subagents else 0
                n_eff = num_effective_subagents[i] if num_effective_subagents else 0
                asr = access_search_ratio[i] if access_search_ratio else 0.0
                if n_total > 0:
                    p_i = min(n_total / max_subagents, 1.0 + epsilon)
                    eff_ratio_i = n_eff / n_total
                    capped = min(asr, 1.0 + epsilon)
                    d = eff_ratio_i + capped
                    c_i = (2.0 * eff_ratio_i * capped / d) if d > 0 else 0.0
                else:
                    p_i = 0.0
                    c_i = 0.0
                reward = parallelism_weight * p_i + completion_weight * c_i
            planner_turn_rewards.append(reward)

        # Scale all turn rewards when total planner turns exceeds soft limit.
        if P > max_planner_turns:
            planner_turn_rewards = [r * max_turn_scale for r in planner_turn_rewards]

        # Map to output_buffer turns.
        turn_rewards = []
        planner_idx = -1
        for turn in output_buffer:
            role = turn.extra_fields.get("role", "")
            if role in ("planner", "single"):
                planner_idx += 1
            if 0 <= planner_idx < P:
                turn_rewards.append(planner_turn_rewards[planner_idx])
            else:
                turn_rewards.append(0.0)

    train_buffer: list[AgentLoopOutput] = list(output_buffer)
    final_answer_format = 1 if answer_format else 0
    return output_buffer, train_buffer, final_answer_format, turn_rewards


async def compute_hind_weights(
    output_buffer: list[AgentLoopOutput],
    extract_answer: str,
    temperature: float = 1.0,
    c_min: float = 0.1,
    c_max: float = 5.0,
    compute_logprobs: Callable[[list[int]], Awaitable[list[float]]] | None = None,
) -> list[float] | None:
    """Compute hindsight importance weights for each non-outcome planner turn.

    For each turn t, builds a hindsight prompt
    ``[state_i,t, "Given that the final outcome was {extract_answer}, how
    necessary was this step?", action_i,t]``, runs it through *compute_logprobs*
    to get token-level logprobs for the action tokens, then computes:

        hind_i,t = exp(sum_logprobs / (temperature × len(action_tokens)))
        rho_i,t  = clip(hind_i,t / mean(hind), c_min, c_max)

    Returns a list of *rho* values, one per planner turn (excluding outcome).
    Returns ``None`` when *compute_logprobs* is not available.

    Args:
        output_buffer: All turns of the trajectory.
        extract_answer: Parsed final answer text.
        temperature: Softmax temperature for logprob weighting.
        c_min, c_max: Clipping bounds for rho.
        compute_logprobs: Async callback (prompt_ids) → per-token logprobs.
    """
    if compute_logprobs is None:
        return None

    # Group turns by planner turn index.
    planner_turns: list[list[AgentLoopOutput]] = []
    current_group: list[AgentLoopOutput] = []
    for t in output_buffer:
        role = t.extra_fields.get("role", "")
        if role in ("planner", "single") and current_group:
            planner_turns.append(current_group)
            current_group = []
        current_group.append(t)
    if current_group:
        planner_turns.append(current_group)

    # The last group is the answer turn — skip it for hind computation.
    if len(planner_turns) <= 1:
        return None

    non_outcome_groups = planner_turns[:-1]
    hind_values: list[float] = []

    for group in non_outcome_groups:
        # Only use the main-agent (planner/single) turn: its prompt already
        # embeds the worker returns from earlier turns, and the action tokens
        # capture the planner's tool-call decisions (including worker creation).
        main_turn = group[0]
        state_text = main_turn.prompt_text or ""
        action_tokens = list(main_turn.response_ids)

        # Build hindsight prompt.
        final_statement = (
            f"Given that the final outcome was {extract_answer}, "
            "what actions do you take in the next step?"
        )
        hindsight_prompt = f"{state_text}\n{final_statement}"

        # Tokenize and compute logprobs.
        # (compute_logprobs handles tokenization internally)
        try:
            logprobs = await compute_logprobs(
                hindsight_prompt, action_tokens, temperature
            )
            s_hind = sum(logprobs)
            n_tokens = len(action_tokens)
            if n_tokens > 0:
                hind = float(
                    __import__("math").exp(s_hind / (temperature * n_tokens))
                )
            else:
                hind = 1.0
        except Exception:
            hind = 1.0

        hind_values.append(hind)

    if not hind_values:
        return None

    # Normalize: rho_i,t = clip(hind_i,t / mean(hind), c_min, c_max)
    mean_hind = sum(hind_values) / len(hind_values)
    if mean_hind <= 0:
        return None

    rho_values = [
        max(c_min, min(c_max, h / mean_hind)) for h in hind_values
    ]
    return rho_values


def _compute_traj_bonuses(
    num_turn_subagents, num_effective_subagents,
    output_buffer, max_subagents, epsilon,
) -> tuple[float, float]:
    """Compute aggregate parallelism and completion bonuses for a trajectory."""
    total_subagents = sum(num_turn_subagents) if num_turn_subagents else 0
    total_effective = sum(num_effective_subagents) if num_effective_subagents else 0
    num_non_outcome = max(len(num_turn_subagents) - 1 if num_turn_subagents else 0, 1)

    parallelism_bonus_val = min(
        total_subagents / (max_subagents * num_non_outcome), 1.0 + epsilon
    )

    total_access = sum(
        t.tool_call_info.get("access", 0)
        for t in output_buffer if t.tool_call_info
    )
    total_search = sum(
        t.tool_call_info.get("search", 0)
        for t in output_buffer if t.tool_call_info
    )
    if total_subagents > 0:
        eff_ratio = total_effective / total_subagents
    else:
        eff_ratio = 0.0
    if total_search > 0:
        acc_ratio = min(total_access / total_search, 1.0 + epsilon)
    else:
        acc_ratio = 0.0
    denom = eff_ratio + acc_ratio
    completion_bonus_val = (
        (2.0 * eff_ratio * acc_ratio / denom) if denom > 0 else 0.0
    )
    return parallelism_bonus_val, completion_bonus_val


async def get_final_reward_score(
    origin_question,
    extract_answer,
    label_answer,
    is_markdown,
    norm_column,
    judge_llm_generator: Callable[[list], Awaitable[str]] | None,
):
    """Compute final reward score for boxed answers or markdown-table answers.

    Args:
        origin_question: Original user question text.
        extract_answer: Parsed model answer (string or DataFrame).
        label_answer: Ground-truth answer payload from dataset.
        is_markdown: Whether to evaluate in markdown-table mode.
        norm_column: Whether to normalize markdown column names aggressively.
        judge_llm_generator: Shared LLM judge generator function backed by SGLang.

    Returns:
        Tuple of `(reward_score, format_ok)`.
    """
    if judge_llm_generator is None:
        return 0.0, False

    format = True
    if is_markdown:
        reward_score, format = await evaluate_markdown(
            extract_answer, label_answer, judge_llm_generator, norm_column
        )
        return reward_score, format

    label_answer = label_answer["answer"]
    if label_answer is not None and extract_answer is not None:
        # Use LLM as judge
        llm_score = await verify_answer_with_llm_judge(
            question=origin_question,
            predicted_answer=extract_answer,
            correct_answer=label_answer,
            judge_llm_generator=judge_llm_generator,
        )

        reward_score = llm_score

    else:
        reward_score = 0.0

    return reward_score, format


async def evaluate_turn_rewards(
    question: str,
    ground_truth: str,
    outcome_reward: float,
    output_buffer: list[AgentLoopOutput],
    judge_llm_generator: Callable[[list], Awaitable[str]],
    num_planner_turns: int,
) -> list[float]:
    """Use an LLM judge to assign per-planner-turn rewards in [-1.0, 1.0].

    The judge sees the full trajectory (planner turns + worker summaries) and
    scores each planner turn based on how much it contributed to the outcome.

    Args:
        question: Original user question.
        ground_truth: Ground-truth answer text.
        outcome_reward: Outcome turn reward (= traj_reward_agg when
            llm_as_turn_rm=True, i.e. llm_reward + bonuses - length_penalty,
            zeroed if no valid answer).
        output_buffer: All turns of the trajectory.
        judge_llm_generator: Shared LLM judge function.
        num_planner_turns: Number of planner turns in this trajectory.

    Returns:
        List of floats, length ``num_planner_turns``, each in [-1.0, 1.0].
    """
    from rlinf.agents.wideseek_r1.utils.prompt import LLM_JUDGE_TURN_PROMPT

    # Build a readable trajectory text.
    parts = []
    turn_idx = 0
    for t in output_buffer:
        role = t.extra_fields.get("role", "")
        if role == "planner":
            turn_idx += 1
            parts.append(f"\n[Turn {turn_idx} - Planner]")
            if t.tool_call_info:
                n = t.tool_call_info.get("subtask", 0)
                parts.append(f"  Created {n} sub-agent(s)")
        elif role == "worker":
            parts.append(f"  Sub-agent response: {t.response_text[:500]}...")

    trajectory_text = "\n".join(parts)

    messages = [
        {"role": "system", "content": "You are an evaluation assistant. For each planner turn, output a score in [-1.0, 1.0] based on how much it helped or hurt progress toward the correct answer."},
        {"role": "user", "content": LLM_JUDGE_TURN_PROMPT.format(
            question=question,
            ground_truth=str(ground_truth),
            outcome_reward=outcome_reward,
            trajectory_text=trajectory_text,
        )},
    ]

    response = await judge_llm_generator(messages)
    response = response.strip()

    # Parse the JSON array from the response.
    try:
        # Try direct JSON parse first.
        scores = json.loads(response)
    except json.JSONDecodeError:
        # Fallback: extract the first JSON array from the text.
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                scores = json.loads(match.group(0))
            except json.JSONDecodeError:
                scores = [0.0] * num_planner_turns
        else:
            scores = [0.0] * num_planner_turns

    # Clamp and pad/truncate to the expected length.
    scores = [max(-1.0, min(1.0, float(s))) for s in scores]
    if len(scores) < num_planner_turns:
        scores.extend([0.0] * (num_planner_turns - len(scores)))
    scores = scores[:num_planner_turns]

    return scores


async def verify_answer_with_llm_judge(
    question: str,
    predicted_answer: str,
    correct_answer: list,
    judge_llm_generator: Callable[[list], Awaitable[str]],
) -> float:
    """Use an LLM judge to score equivalence between prediction and reference.

    Args:
        question: Original user question.
        predicted_answer: Model-predicted boxed answer.
        correct_answer: Reference answer list from dataset.
        judge_llm_generator: Shared LLM judge generator function backed by SGLang.

    Returns:
        `1.0` if judged correct, otherwise `0.0`.
    """
    from rlinf.agents.wideseek_r1.utils.prompt import LLM_JUDGE_PROMPT

    if len(correct_answer) == 1:
        # Format the judge prompt
        judge_prompt_text = LLM_JUDGE_PROMPT.format(
            question=question,
            correct_answer=correct_answer[0],
            response=predicted_answer,
        )
    else:
        judge_prompt_text = LLM_JUDGE_PROMPT.format(
            question=question, correct_answer=correct_answer, response=predicted_answer
        )

    judge_messages = [
        {
            "role": "system",
            "content": "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.",
        },
        {"role": "user", "content": judge_prompt_text},
    ]
    # Use provided judge_llm_generator function to get judge response
    judge_response_text = await judge_llm_generator(judge_messages)

    judge_response_clean = judge_response_text.strip().lower()
    if "correct" in judge_response_clean and "incorrect" not in judge_response_clean:
        return 1.0
    else:
        return 0.0


async def evaluate_markdown(
    extract_answer,
    label_answer,
    judge_llm_generator: Callable[[list], Awaitable[str]],
    norm_column_=False,
):
    """Evaluate markdown-table answers with schema checks and LLM cell matching.

    Args:
        extract_answer: Parsed prediction DataFrame.
        label_answer: Ground-truth markdown payload or DataFrame.
        judge_llm_generator: Shared LLM judge generator function backed by SGLang.
        norm_column_: Whether to normalize spaces in column names.

    Returns:
        Tuple of `(score, format_ok)` where `score` is item-level F1.
    """

    # Helper function to normalize column names
    def norm_column(col: str) -> str:
        """Normalize column names to improve schema alignment robustness."""
        if not norm_column_:
            return col.strip().lower()
        else:
            return col.strip().lower().replace(" ", "")

    # Helper function to calculate F1 score
    def calc_f1(precision, recall):
        """Compute a numerically stable F1 score."""
        epsilon = 1e-9
        return (
            (2 * precision * recall / (precision + recall))
            if (precision + recall > epsilon)
            else 0.0
        )

    def normalize_series_to_str(s: pd.Series) -> pd.Series:
        """Normalize a series to stripped canonical strings for matching."""
        s0 = s.astype(str).str.strip()
        num = pd.to_numeric(s0, errors="coerce")
        if num.notna().any():
            return num.map(lambda x: "" if pd.isna(x) else f"{x:g}")
        else:
            return s0

    # Initialize metrics
    precision_by_item = 0.0
    recall_by_item = 0.0
    f1_by_item = 0.0

    try:
        # Parse label_answer
        if isinstance(label_answer, dict):
            answer_markdown = label_answer.get("answer", "")
            unique_columns = label_answer.get("unique_columns", [])
            required_columns = label_answer.get("required", [])
        else:
            # If label_answer is a string, assume it's markdown
            answer_markdown = label_answer
            unique_columns = []
            required_columns = []

        # Convert answer_markdown to DataFrame if it's a string
        if isinstance(answer_markdown, str):
            answer_df = extract_final_answer(
                answer_markdown, mode="markdown", strict=False
            )
            if answer_df is None:
                # print("Failed to parse label answer markdown")
                return 0.0, False
        elif isinstance(answer_markdown, pd.DataFrame):
            answer_df = answer_markdown
        else:
            # print(f"Invalid label answer type: {type(answer_markdown)}")
            return 0.0, False

        # Validate extract_answer
        if not isinstance(extract_answer, pd.DataFrame) or extract_answer.empty:
            # print(f"Extracted answer is None or not a DataFrame, it's {extract_answer}")
            return 0.0, False

        response_df = copy.deepcopy(extract_answer)

        # Normalize column names
        answer_df.columns = [norm_column(col) for col in answer_df.columns]
        response_df.columns = [norm_column(col) for col in response_df.columns]

        # Normalize unique_columns and required_columns
        unique_columns = [norm_column(col) for col in unique_columns]

        if not required_columns:
            required_columns = list(answer_df.columns)
        else:
            required_columns = [
                norm_column(col) for col in required_columns
            ]  # widesearch requir columns: " " -> ""

        # Check if response has required columns
        if not set(required_columns).issubset(set(response_df.columns)):
            # Try primary key preprocessing to map column names
            column_map = await primary_key_preprocess(
                list(response_df.columns),
                required_columns,
                judge_llm_generator,
            )
            response_df.rename(columns=column_map, inplace=True)

        if not set(required_columns).issubset(set(response_df.columns)):
            # print(f"required_columns {required_columns} != response_df {list(response_df.columns)}")
            return 0.0, False

        for col in required_columns:
            answer_df[col] = normalize_series_to_str(answer_df[col])
            response_df[col] = normalize_series_to_str(response_df[col])

        # Remove duplicates based on unique columns
        if unique_columns:
            response_df.drop_duplicates(subset=unique_columns, inplace=True)
            answer_df.drop_duplicates(subset=unique_columns, inplace=True)

            # Preprocess primary keys using LLM
            for col in unique_columns:
                primary_key_map = await primary_key_preprocess(
                    response_df[col].tolist(),
                    answer_df[col].tolist(),
                    judge_llm_generator,
                )
                response_df[col + "_before_map"] = response_df[col]
                response_df[col] = response_df[col].apply(
                    lambda x: primary_key_map.get(x, x)
                )

        # Inner join over unique keys to align comparable rows.
        df_inner = pd.merge(
            answer_df,
            response_df,
            on=unique_columns,
            how="inner",
            suffixes=("_query", "_response"),
        )

        # Initialize score DataFrames for each metric type in reward_eval
        df_inner_scores = pd.DataFrame(index=df_inner.index)

        llm_tasks = []
        llm_columns = []

        # Process each column
        for col in required_columns:
            if col in unique_columns:
                df_inner_scores[f"{col}_score"] = 1.0
            else:
                responses = df_inner[col + "_response"].tolist()
                targets = df_inner[col + "_query"].tolist()
                llm_tasks.append(
                    llm_judge_column(responses, targets, judge_llm_generator)
                )
                llm_columns.append(col)

        # Execute LLM semantic checks in parallel per non-key column.
        if llm_tasks:
            llm_results = await asyncio.gather(*llm_tasks)
            # Assign results back to df_inner_scores["LLM"]
            for col, scores in zip(llm_columns, llm_results):
                df_inner_scores[f"{col}_score"] = scores

        # Calculate metrics for each evaluation method
        num_pred_rows = len(response_df)
        num_gt_rows = len(answer_df)
        num_pred_items = num_pred_rows * len(required_columns)
        num_gt_items = num_gt_rows * len(required_columns)

        # Item-level metrics
        tp_by_item = df_inner_scores.sum().sum()
        precision_by_item = tp_by_item / num_pred_items if num_pred_items > 0 else 0.0
        recall_by_item = tp_by_item / num_gt_items if num_gt_items > 0 else 0.0
        f1_by_item = calc_f1(precision_by_item, recall_by_item)

    except Exception:
        # print(f"Evaluation error: {traceback.format_exc()}")
        return 0.0, False

    return f1_by_item, True


async def llm_judge_column(
    responses: list,
    targets: list,
    judge_llm_generator: Callable[[list], Awaitable[str]],
) -> list:
    """Score non-key markdown table cells using semantic LLM comparison.

    Args:
        responses: Predicted cell values for one column.
        targets: Ground-truth cell values for one column.
        judge_llm_generator: Shared LLM judge generator function backed by SGLang.

    Returns:
        List of float scores aligned with `responses`.
    """
    criterion = "It is sufficient if the semantics are approximately the same as the reference answer or if they point to the same entity. There is no need for a word-for-word correspondence."

    if not responses:
        return []

    # Widesearch's eval_column_prompt
    eval_column_prompt = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by step according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
"idx_0": score,
"idx_1": score,
...
}}
```

{criterion}
"""

    user_prompt = """Here is the response you need to judge, please make sure to analyze each item step by step before providing the final scoring results.

{response}
"""

    # Build response dict
    response_dict = {}
    for idx, (resp, tar) in enumerate(zip(responses, targets)):
        response_dict[f"idx_{idx}"] = {"response": str(resp), "target": str(tar)}

    # Format prompt
    system_prompt = eval_column_prompt.format(
        criterion=criterion,
    )

    user_prompt = user_prompt.format(response=response_dict)
    # Create messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Use provided judge_llm_generator function to get judge response
    result_text = await judge_llm_generator(messages)

    try:
        pat = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pat, result_text, re.DOTALL)
        if matches:
            json_str = matches[-1]
            score_dict = json.loads(json_str)
            score_list = [
                float(score_dict.get(f"idx_{idx}", 0)) for idx in range(len(responses))
            ]
        else:
            # Parsing failed, default to 0
            score_list = [0.0] * len(responses)
    except Exception:
        # If any error, default to 0
        score_list = [0.0] * len(responses)

    # Ensure correct length
    if len(score_list) != len(responses):
        score_list = [0.0] * len(responses)
    return score_list


async def primary_key_preprocess(
    response_list, reference_list, judge_llm_generator: Callable[[list], Awaitable[str]]
):
    """Align predicted primary-key values to reference canonical forms.

    Args:
        response_list: Candidate values from prediction side.
        reference_list: Reference values used as canonical vocabulary.
        judge_llm_generator: Shared LLM judge generator function backed by SGLang.

    Returns:
        Mapping from predicted string to aligned reference string.
    """
    primary_key_map = {}

    # The prompt template from widesearch
    primary_key_preprocess_prompt = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
"origin_str1": "transform_str1",
"origin_str2": "transform_str2"
}}
```
"""

    user_prompt = """
The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}
"""

    # Format the prompt
    system_prompt = primary_key_preprocess_prompt

    user_prompt = user_prompt.format(response=response_list, reference=reference_list)

    # Create messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Use provided judge_llm_generator function to get judge response
    result_text = await judge_llm_generator(messages)

    # Parse JSON from result
    try:
        pat = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pat, result_text, re.DOTALL)
        if matches:
            json_str = matches[-1]
            transform_map = json.loads(json_str)
            primary_key_map.update(transform_map)
    except Exception:
        pass

    return primary_key_map


def extract_final_answer(text: str, mode: bool = "boxed", strict=True):
    """Extract final answer from generated text using a specific parsing mode.

    Args:
        text: Raw generated text that may include reasoning/tool wrappers.
        mode: Parsing mode (`tag`, `boxed`, or `markdown`).
        strict: For markdown mode, require fenced markdown blocks when True.

    Returns:
        For `tag`/`boxed`: extracted string or None.
        For `markdown`: parsed `pd.DataFrame` or None.
    """
    text = text.split("</think>")[-1].strip()

    # When <answer> tags are present, extract their content first.
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_matches = re.findall(answer_pattern, text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1].strip()

    if mode == "tag":
        return text if text else None
    elif mode == "boxed":
        if not text:
            return None

        matches = []
        i = 0

        while i < len(text):
            boxed_start = text.find(r"\boxed{", i)
            if boxed_start == -1:
                break

            content_start = boxed_start + 7  # len(r'\boxed{') = 7
            if content_start >= len(text):
                break

            # Count balanced braces
            brace_count = 1
            content_end = content_start

            while content_end < len(text) and brace_count > 0:
                char = text[content_end]
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                content_end += 1

            if brace_count == 0:
                content = text[content_start : content_end - 1]
                matches.append(content)
                i = content_end
            else:
                i = content_start

        return matches[-1] if matches else None
    elif mode == "markdown":
        if not text or not isinstance(text, str):
            return None

        response_df = None
        markdown_str = re.findall(r"```markdown(.*?)```", text, re.DOTALL)
        if not markdown_str:
            # Fallback parser for answers that forgot markdown fences.
            if strict:
                return None
            pipe_positions = [m.start() for m in re.finditer(r"\|", text)]
            if len(pipe_positions) >= 4:
                first_pipe = pipe_positions[0]
                last_pipe = pipe_positions[-1]
                start = text.rfind("\n", 0, first_pipe)
                start = 0 if start == -1 else start
                end = text.find("\n", last_pipe)
                end = len(text) if end == -1 else end
                table_candidate = text[start:end]
                markdown_str = re.findall(r"((?:\|.*\n?)+)", table_candidate)
        if markdown_str:
            markdown_str = markdown_str[-1].strip()
            lines = markdown_str.split("\n")
            # lines[0] = lines[0].replace(" ", "").lower()
            lines = [line.strip() for line in lines]

            new_lines = []
            for line in lines:
                if set(line.strip()).issubset(set("|- :")) or "|" not in line:
                    continue
                new_lines.append("|".join([_line.strip() for _line in line.split("|")]))

            if not new_lines:
                return None
            markdown_str = "\n".join(new_lines)
            try:
                response_df = pd.read_csv(
                    StringIO(markdown_str), sep="|", keep_default_na=False
                )
                response_df = response_df.loc[
                    :, ~response_df.columns.str.startswith("Unnamed")
                ]

                for col in response_df.columns:  # FIXME: check if need？
                    if response_df[col].dtype == "object":
                        response_df[col] = response_df[col].apply(
                            lambda x: (
                                x.replace("<br>", "\n")
                                if isinstance(x, str) and x
                                else x
                            )
                        )
                    response_df[col] = response_df[col].replace("", "nan")

                return response_df
            except Exception:
                # print(f"Error parsing markdown table: {e}")
                return None

        return response_df
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'tag', 'boxed', or 'markdown'")
