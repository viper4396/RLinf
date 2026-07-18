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
    answer_mode,
    norm_column,
    judge_llm_generator: Callable[[list], Awaitable[str]] | None,
):
    """Compute final reward score for boxed answers or markdown-table answers.

    Args:
        origin_question: Original user question text.
        extract_answer: Parsed model answer (string or DataFrame).
        label_answer: Ground-truth answer payload from dataset.
        answer_mode: Answer mode for this sample (``markdown`` or ``boxed``).
        norm_column: Whether to normalize markdown column names aggressively.
        judge_llm_generator: Shared LLM judge generator function backed by SGLang.
            GISA exact-match evaluation does not use it.

    Returns:
        Tuple of `(reward_score, format_ok)`.
    """
    if isinstance(label_answer, dict) and label_answer.get("is_gisa", False):
        return evaluate_gisa_exact_match(
            extract_answer=extract_answer,
            label_answer=label_answer,
        )

    if judge_llm_generator is None:
        return 0.0, False

    if answer_mode == "markdown":
        return await evaluate_markdown(
            extract_answer, label_answer, judge_llm_generator, norm_column
        )

    label_answer = label_answer["answer"]
    if label_answer is not None and extract_answer is not None:
        # Use LLM as judge
        reward_score = await verify_answer_with_llm_judge(
            question=origin_question,
            predicted_answer=extract_answer,
            correct_answer=label_answer,
            judge_llm_generator=judge_llm_generator,
        )
    else:
        reward_score = 0.0

    return reward_score, True


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
    from rlinf.agents.wideseek_r2.utils.prompt import LLM_JUDGE_PROMPT

    # A single-element reference list is unwrapped to its only element.
    reference = correct_answer[0] if len(correct_answer) == 1 else correct_answer
    judge_prompt_text = LLM_JUDGE_PROMPT.format(
        question=question,
        correct_answer=reference,
        response=predicted_answer,
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


def _markdown_dataframe(answer) -> pd.DataFrame | None:
    """Return a parsed Markdown DataFrame, accepting reference strings."""
    if isinstance(answer, str):
        answer = extract_final_answer(answer, mode="markdown", strict=False)
    if not isinstance(answer, pd.DataFrame) or answer.empty:
        return None
    return answer


def _normalize_em_value(value) -> str:
    """Normalize surrounding whitespace while preserving exact text."""
    return "" if pd.isna(value) else str(value).strip()


def _markdown_contents(answer) -> list | None:
    """Return Markdown-table row contents without exposing column names."""
    answer_df = _markdown_dataframe(answer)
    if answer_df is None:
        return None

    contents = []
    for row in answer_df.itertuples(index=False, name=None):
        normalized_row = [_normalize_em_value(value) for value in row]
        contents.append(
            normalized_row[0] if len(normalized_row) == 1 else normalized_row
        )
    return contents


def _markdown_table_signature(answer) -> tuple | None:
    """Return exact ordered headers and rows for a Markdown table."""
    answer_df = _markdown_dataframe(answer)
    if answer_df is None:
        return None
    columns = tuple(_normalize_em_value(column) for column in answer_df.columns)
    rows = tuple(
        tuple(_normalize_em_value(value) for value in row)
        for row in answer_df.itertuples(index=False, name=None)
    )
    return columns, rows


def _freeze_content(value):
    """Convert nested row content into a hashable exact-match value."""
    if isinstance(value, list):
        return tuple(_freeze_content(item) for item in value)
    return value


def evaluate_gisa_exact_match(
    extract_answer,
    label_answer: dict,
) -> tuple[float, bool]:
    """Evaluate every GISA answer locally with deterministic exact match.

    Item answers use direct string EM. Markdown tables compare ordered headers
    and rows. Sets and lists discard headers and compare only content rows;
    sets ignore order while lists preserve it.

    Args:
        extract_answer: Parsed prediction DataFrame.
        label_answer: Ground-truth answer payload.

    Returns:
        ``(score, format_ok)`` with a deterministic binary EM score.
    """
    answer_type = label_answer.get("answer_type")
    answer_mode = label_answer.get("answer_mode")

    if answer_mode == "boxed" and answer_type == "item":
        if extract_answer is None:
            return 0.0, False
        references = label_answer.get("answer", [])
        if not isinstance(references, list):
            references = [references]
        prediction = _normalize_em_value(extract_answer)
        score = any(
            prediction == _normalize_em_value(reference) for reference in references
        )
        return float(score), True

    if answer_mode != "markdown":
        return 0.0, False

    correct_answer = label_answer.get("answer", "")
    if answer_type == "table":
        correct_table = _markdown_table_signature(correct_answer)
        predicted_table = _markdown_table_signature(extract_answer)
        if correct_table is None or predicted_table is None:
            return 0.0, False
        return float(predicted_table == correct_table), True

    if answer_type not in {"set", "list"}:
        return 0.0, False

    correct_contents = _markdown_contents(correct_answer)
    predicted_contents = _markdown_contents(extract_answer)
    if correct_contents is None or predicted_contents is None:
        return 0.0, False

    if answer_type == "set":
        correct_values = {_freeze_content(value) for value in correct_contents}
        predicted_values = {_freeze_content(value) for value in predicted_contents}
        return float(predicted_values == correct_values), True

    return float(predicted_contents == correct_contents), True


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


def extract_final_answer(text: str, mode: str = "boxed", strict=True):
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
    if mode == "tag":
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.finditer(answer_pattern, text, re.DOTALL)
        matches = list(match)

        if len(matches) < 1:
            return None
        return matches[-1].group(1).strip()
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
