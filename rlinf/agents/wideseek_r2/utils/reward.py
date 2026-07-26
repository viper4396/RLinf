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


def _extract_json_object(text: str) -> dict | None:
    """Extract the first valid JSON object from a judge response."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates.extend(re.findall(r"\{[^{}]*\}", text, re.DOTALL))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


async def evaluate_worker_quality(
    main_question: str,
    subtask: str,
    worker_summary: str | None,
    evidence_context: str,
    judge_llm_generator: Callable[[list], Awaitable[str]] | None,
) -> tuple[float, bool]:
    """Judge one worker's grounded subtask quality on a ``[0, 1]`` scale.

    Args:
        main_question: Original user question.
        subtask: Subtask assigned by the planner.
        worker_summary: Worker's final non-empty ``<answer>`` content.
        evidence_context: Search/access context visible to the worker.
        judge_llm_generator: Shared LLM judge callback.

    Returns:
        ``(quality, valid)``. ``valid`` is false when no usable judge score is
        available, allowing the advantage code to substitute a neutral baseline.
    """
    if not worker_summary or not worker_summary.strip() or judge_llm_generator is None:
        return 0.0, False

    judge_messages = [
        {
            "role": "system",
            "content": (
                "You evaluate a research sub-agent. Score only information supported "
                "by its evidence. Return JSON with numeric fields relevance, "
                "groundedness, coverage, and usefulness, each between 0 and 1."
            ),
        },
        {
            "role": "user",
            "content": (
                f"MAIN QUESTION:\n{main_question}\n\n"
                f"ASSIGNED SUBTASK:\n{subtask}\n\n"
                f"EVIDENCE AND WORKER CONTEXT:\n{evidence_context}\n\n"
                f"WORKER SUMMARY:\n{worker_summary}\n\n"
                "Return JSON only."
            ),
        },
    ]
    try:
        response = await judge_llm_generator(judge_messages)
        scores = _extract_json_object(response)
        if scores is None:
            return 0.0, False
        weights = {
            "relevance": 0.25,
            "groundedness": 0.30,
            "coverage": 0.25,
            "usefulness": 0.20,
        }
        quality = 0.0
        for name, weight in weights.items():
            value = float(scores[name])
            quality += weight * max(0.0, min(1.0, value))
        return quality, True
    except Exception:
        return 0.0, False


def credit_assignment(
    agentloop_config: DictConfig,
    llm_reward,
    answer_format,
):
    """Compute the trajectory reward from the answer score and format reward.

    The reward is intentionally simple: the answer score plus a format bonus
    that is granted only when the final answer was extracted with a valid format.

    Args:
        agentloop_config: Agent-loop config containing the ``format_reward`` weight.
        llm_reward: End-of-trajectory reward from answer evaluation.
        answer_format: Whether final-answer extraction/format validation succeeded.

    Returns:
        The scalar ``reward_score`` shared by every turn in the trajectory.
    """
    format_reward = agentloop_config.get("format_reward", 0.0)
    return llm_reward + (format_reward if answer_format else 0.0)


async def get_final_reward_score(
    origin_question,
    extract_answer,
    label_answer,
    norm_column,
    judge_llm_generator: Callable[[list], Awaitable[str]] | None,
):
    """Compute the final reward score for a Markdown-table answer.

    Args:
        origin_question: Original user question text.
        extract_answer: Parsed model answer DataFrame.
        label_answer: Ground-truth answer payload from dataset.
        norm_column: Whether to normalize markdown column names aggressively.
        judge_llm_generator: Shared LLM judge generator function backed by SGLang.

    Returns:
        Tuple of ``(reward_score, format_ok, answer_metrics)``. GISA samples
        populate ``answer_metrics`` with judge-based structured scores.
    """
    if isinstance(label_answer, dict) and label_answer.get("is_gisa", False):
        return await evaluate_gisa_with_llm_judge(
            question=origin_question,
            extract_answer=extract_answer,
            label_answer=label_answer,
            judge_llm_generator=judge_llm_generator,
        )

    if judge_llm_generator is None:
        return 0.0, False, {}

    if label_answer.get("answer_type") == "item":
        predicted_item = _markdown_item_value(extract_answer)
        if predicted_item is None:
            return 0.0, False, {}
        reward_score = await verify_answer_with_llm_judge(
            question=origin_question,
            predicted_answer=predicted_item,
            correct_answer=label_answer.get("answer"),
            judge_llm_generator=judge_llm_generator,
        )
        return reward_score, True, {}

    reward_score, format_ok = await evaluate_markdown(
        extract_answer, label_answer, judge_llm_generator, norm_column
    )
    return reward_score, format_ok, {}


async def verify_answer_with_llm_judge(
    question: str,
    predicted_answer,
    correct_answer,
    judge_llm_generator: Callable[[list], Awaitable[str]],
    answer_type: str = "item",
) -> float:
    """Use an LLM judge to score equivalence between prediction and reference.

    Args:
        question: Original user question.
        predicted_answer: Model-predicted answer.
        correct_answer: Reference answer or accepted-reference list.
        judge_llm_generator: Shared LLM judge generator function backed by SGLang.
        answer_type: Structural semantics used when comparing the answers.

    Returns:
        `1.0` if judged correct, otherwise `0.0`.
    """
    from rlinf.agents.wideseek_r2.utils.prompt import LLM_JUDGE_PROMPT

    # A single-element reference list is unwrapped to its only element.
    reference = (
        correct_answer[0]
        if isinstance(correct_answer, list) and len(correct_answer) == 1
        else correct_answer
    )
    judge_prompt_text = LLM_JUDGE_PROMPT.format(
        question=question,
        correct_answer=reference,
        response=predicted_answer,
    )

    type_rules = {
        "item": "The prediction passes when it is semantically equivalent to any accepted reference answer.",
        "set": "Treat both answers as sets: ignore member order and duplicate occurrences, but require the same members.",
        "list": "Treat both answers as ordered lists: require the same members in the same order, including meaningful duplicates.",
        "table": "Treat both answers as tables: require the requested schema and semantically equivalent row and cell content.",
    }
    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are an evaluation assistant. Determine whether the predicted "
                "answer is equivalent to the labeled answer. "
                f"{type_rules.get(answer_type, type_rules['table'])} "
                "Allow harmless wording and formatting differences. Conclude with "
                "exactly Correct or Incorrect."
            ),
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


def _markdown_dataframe(answer) -> pd.DataFrame | None:
    """Return a parsed Markdown DataFrame, accepting reference strings."""
    if isinstance(answer, str):
        answer = extract_final_answer(answer, mode="markdown", strict=False)
    elif isinstance(answer, list):
        try:
            if answer and all(
                not isinstance(value, (dict, list, tuple)) for value in answer
            ):
                answer = pd.DataFrame({"Item": answer})
            else:
                answer = pd.DataFrame.from_records(answer)
        except (TypeError, ValueError):
            return None
    if not isinstance(answer, pd.DataFrame) or answer.empty:
        return None
    return answer


def _normalize_answer_value(value) -> str:
    """Normalize an answer value for judge serialization."""
    return "" if pd.isna(value) else str(value).strip()


def _markdown_item_value(answer) -> str | None:
    """Return the sole value from a valid one-row ``Item`` Markdown table."""
    answer_df = _markdown_dataframe(answer)
    if answer_df is None or answer_df.shape != (1, 1):
        return None
    if _normalize_answer_value(answer_df.columns[0]).lower() != "item":
        return None
    return _normalize_answer_value(answer_df.iat[0, 0])


def _f1_score(matches: int, predictions: int, references: int) -> float:
    """Calculate F1 from a number of semantically matched items."""
    denominator = predictions + references
    return 2.0 * matches / denominator if denominator else 0.0


def _maximum_semantic_matching(matrix: list[list[bool]]) -> list[tuple[int, int]]:
    """Return a maximum one-to-one matching from a semantic-equivalence matrix."""
    if not matrix or not matrix[0]:
        return []

    reference_matches = [-1] * len(matrix[0])

    def augment(prediction_index: int, seen: list[bool]) -> bool:
        for reference_index, equivalent in enumerate(matrix[prediction_index]):
            if not equivalent or seen[reference_index]:
                continue
            seen[reference_index] = True
            previous_prediction = reference_matches[reference_index]
            if previous_prediction < 0 or augment(previous_prediction, seen):
                reference_matches[reference_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(matrix)):
        augment(prediction_index, [False] * len(matrix[0]))

    return [
        (prediction_index, reference_index)
        for reference_index, prediction_index in enumerate(reference_matches)
        if prediction_index >= 0
    ]


async def _semantic_equivalence_matrix(
    predictions: list,
    references: list,
    judge_llm_generator: Callable[[list], Awaitable[str]],
) -> list[list[bool]]:
    """Judge every prediction/reference pair and return an equivalence matrix."""
    if not predictions or not references:
        return [[False] * len(references) for _ in predictions]

    pair_predictions = []
    pair_references = []
    for prediction in predictions:
        for reference in references:
            pair_predictions.append(prediction)
            pair_references.append(reference)

    scores = await llm_judge_column(
        pair_predictions,
        pair_references,
        judge_llm_generator,
    )
    width = len(references)
    return [
        [score >= 0.5 for score in scores[offset : offset + width]]
        for offset in range(0, len(scores), width)
    ]


def _semantic_order_score(matrix: list[list[bool]]) -> float:
    """Calculate an LCS-style order score from semantic element matches."""
    if not matrix or not matrix[0]:
        return 0.0

    num_predictions = len(matrix)
    num_references = len(matrix[0])
    lcs = [[0] * (num_references + 1) for _ in range(num_predictions + 1)]
    for prediction_index in range(1, num_predictions + 1):
        for reference_index in range(1, num_references + 1):
            if matrix[prediction_index - 1][reference_index - 1]:
                lcs[prediction_index][reference_index] = (
                    lcs[prediction_index - 1][reference_index - 1] + 1
                )
            else:
                lcs[prediction_index][reference_index] = max(
                    lcs[prediction_index - 1][reference_index],
                    lcs[prediction_index][reference_index - 1],
                )
    return _f1_score(
        lcs[num_predictions][num_references],
        num_predictions,
        num_references,
    )


async def _evaluate_gisa_collection(
    answer_type: str,
    predicted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    judge_llm_generator: Callable[[list], Awaitable[str]],
) -> dict[str, float]:
    """Evaluate a set or list with semantic cell matching."""
    predicted_values = [
        _normalize_answer_value(value) for value in predicted_df.iloc[:, 0].tolist()
    ]
    reference_values = [
        _normalize_answer_value(value) for value in reference_df.iloc[:, 0].tolist()
    ]
    if answer_type == "set":
        predicted_values = list(dict.fromkeys(predicted_values))
        reference_values = list(dict.fromkeys(reference_values))

    matrix = await _semantic_equivalence_matrix(
        predicted_values,
        reference_values,
        judge_llm_generator,
    )
    matched_items = len(_maximum_semantic_matching(matrix))
    cell_f1 = _f1_score(
        matched_items,
        len(predicted_values),
        len(reference_values),
    )
    metrics = {"cell_f1": cell_f1}
    if answer_type == "list":
        order_score = _semantic_order_score(matrix)
        metrics["order_score"] = order_score
        metrics["pass"] = float(cell_f1 == 1.0 and order_score == 1.0)
    else:
        metrics["pass"] = float(cell_f1 == 1.0)
    return metrics


def _normalized_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return a copy with normalized, unique column names."""
    result = df.copy()
    result.columns = [
        _normalize_answer_value(column).lower() for column in result.columns
    ]
    if result.columns.has_duplicates:
        return None
    return result


def _table_rows(df: pd.DataFrame, columns: list[str]) -> list[dict[str, str]]:
    """Serialize selected table columns into row dictionaries for the judge."""
    return [
        {
            column: _normalize_answer_value(df.iloc[row_index][column])
            for column in columns
        }
        for row_index in range(len(df))
    ]


async def _evaluate_gisa_table(
    predicted_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    unique_columns: list,
    judge_llm_generator: Callable[[list], Awaitable[str]],
) -> dict[str, float]:
    """Evaluate table cells and complete rows with semantic LLM matching."""
    predicted_df = _normalized_table(predicted_df)
    reference_df = _normalized_table(reference_df)
    if predicted_df is None or reference_df is None:
        return {"cell_f1": 0.0, "row_f1": 0.0, "pass": 0.0}

    shared_columns = [
        column for column in reference_df.columns if column in predicted_df.columns
    ]
    num_predictions = predicted_df.size
    num_references = reference_df.size
    if not shared_columns:
        return {"cell_f1": 0.0, "row_f1": 0.0, "pass": 0.0}

    normalized_unique_columns = [
        _normalize_answer_value(column).lower() for column in (unique_columns or [])
    ]
    if normalized_unique_columns and set(normalized_unique_columns).issubset(
        set(shared_columns)
    ):
        prediction_keys = _table_rows(predicted_df, normalized_unique_columns)
        reference_keys = _table_rows(reference_df, normalized_unique_columns)
        key_matrix = await _semantic_equivalence_matrix(
            prediction_keys,
            reference_keys,
            judge_llm_generator,
        )
        aligned_rows = _maximum_semantic_matching(key_matrix)
    else:
        aligned_rows = [
            (row_index, row_index)
            for row_index in range(min(len(predicted_df), len(reference_df)))
        ]

    cell_predictions = []
    cell_references = []
    for prediction_index, reference_index in aligned_rows:
        for column in shared_columns:
            cell_predictions.append(predicted_df.iloc[prediction_index][column])
            cell_references.append(reference_df.iloc[reference_index][column])

    cell_scores = await llm_judge_column(
        cell_predictions,
        cell_references,
        judge_llm_generator,
    )
    matched_cells = sum(score >= 0.5 for score in cell_scores)
    cell_f1 = _f1_score(matched_cells, num_predictions, num_references)

    predicted_rows = _table_rows(predicted_df, shared_columns)
    reference_rows = _table_rows(reference_df, shared_columns)
    row_matrix = await _semantic_equivalence_matrix(
        predicted_rows,
        reference_rows,
        judge_llm_generator,
    )
    matched_rows = len(_maximum_semantic_matching(row_matrix))
    row_f1 = _f1_score(matched_rows, len(predicted_rows), len(reference_rows))
    return {
        "cell_f1": cell_f1,
        "row_f1": row_f1,
        "pass": float(cell_f1 == 1.0 and row_f1 == 1.0),
    }


async def evaluate_gisa_with_llm_judge(
    question: str,
    extract_answer,
    label_answer: dict,
    judge_llm_generator: Callable[[list], Awaitable[str]] | None,
) -> tuple[float, bool, dict[str, float]]:
    """Evaluate GISA cells and rows with semantic LLM-judge decisions.

    Args:
        question: Original user question.
        extract_answer: Parsed prediction DataFrame.
        label_answer: Ground-truth answer payload.
        judge_llm_generator: Shared LLM judge callback.

    Returns:
        ``(cell_f1, format_ok, metrics)``. Metrics always include judge-based
        ``cell_f1`` and ``pass`` and may include ``row_f1`` or ``order_score``.
    """
    if judge_llm_generator is None:
        return 0.0, False, {}

    answer_type = label_answer.get("answer_type", "table")
    answer_df = _markdown_dataframe(extract_answer)
    if answer_df is None:
        return 0.0, False, {}
    if answer_type == "item":
        predicted_item = _markdown_item_value(answer_df)
        if predicted_item is None:
            return 0.0, False, {}
        cell_f1 = await verify_answer_with_llm_judge(
            question=question,
            predicted_answer=predicted_item,
            correct_answer=label_answer.get("answer"),
            judge_llm_generator=judge_llm_generator,
            answer_type=answer_type,
        )
        metrics = {"cell_f1": cell_f1, "pass": cell_f1}
    elif answer_type in {"set", "list"}:
        if answer_df.shape[1] != 1:
            return 0.0, False, {}
        if _normalize_answer_value(answer_df.columns[0]).lower() != "item":
            return 0.0, False, {}
        reference_df = _markdown_dataframe(label_answer.get("answer"))
        if reference_df is None or reference_df.shape[1] != 1:
            return 0.0, False, {}
        metrics = await _evaluate_gisa_collection(
            answer_type,
            answer_df,
            reference_df,
            judge_llm_generator,
        )
    elif answer_type == "table":
        reference_df = _markdown_dataframe(label_answer.get("answer"))
        if reference_df is None:
            return 0.0, False, {}
        metrics = await _evaluate_gisa_table(
            answer_df,
            reference_df,
            label_answer.get("unique_columns", []),
            judge_llm_generator,
        )
    elif answer_type != "table":
        return 0.0, False, {}

    return metrics["cell_f1"], True, metrics


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
    user_prompt = user_prompt.format(response=response_list, reference=reference_list)

    # Create messages
    messages = [
        {"role": "system", "content": primary_key_preprocess_prompt},
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


def extract_final_answer(text: str, mode: str = "markdown", strict=True):
    """Extract final answer from generated text using a specific parsing mode.

    Args:
        text: Raw generated text that may include reasoning/tool wrappers.
        mode: Parsing mode (``tag`` for workers or ``markdown`` for main roles).
        strict: For markdown mode, require fenced markdown blocks when True.

    Returns:
        For ``tag``: extracted string or None.
        For ``markdown``: parsed ``pd.DataFrame`` or None.
    """
    text = text.split("</think>")[-1].strip()
    if mode == "tag":
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.finditer(answer_pattern, text, re.DOTALL)
        matches = list(match)

        if len(matches) < 1:
            return None
        return matches[-1].group(1).strip()
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
        raise ValueError(f"Unknown mode: {mode}. Must be 'tag' or 'markdown'")
