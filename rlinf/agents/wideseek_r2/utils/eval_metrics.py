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

from rlinf.agents.wideseek_r2.utils.reward import evaluate_gisa_markdown_scores

GISA_MARKDOWN_ANSWER_TYPES = frozenset({"table", "set", "list"})


def _aggregate_grouped_scores(
    score_groups: list[list[float]], metric_name: str
) -> dict[str, float]:
    """Aggregate first, average, and maximum rollout scores by question."""
    if not score_groups:
        return {}
    first_scores = [scores[0] for scores in score_groups]
    average_scores = [sum(scores) / len(scores) for scores in score_groups]
    maximum_scores = [max(scores) for scores in score_groups]
    return {
        f"{metric_name}@1": sum(first_scores) / len(first_scores),
        f"avg_{metric_name}@k": sum(average_scores) / len(average_scores),
        f"max_{metric_name}@k": sum(maximum_scores) / len(maximum_scores),
    }


def aggregate_gisa_markdown_metrics(
    raw_results: list[dict], *, enabled: bool
) -> dict[str, float]:
    """Aggregate GISA Markdown EM and answer-type-specific metrics.

    The aggregation includes universal whole-answer EM, table row F1, and list
    order score. Existing cell-F1 aggregation remains in the evaluation runner.

    Args:
        raw_results: Per-question rollout results produced by the evaluator.
        enabled: Whether GISA Markdown evaluation is enabled.

    Returns:
        GISA Markdown metrics, or an empty dictionary when disabled.
    """
    if not enabled:
        return {}

    exact_match_groups = []
    row_f1_groups = []
    order_score_groups = []
    for raw_result in raw_results:
        answer = raw_result.get("answer")
        if not (
            isinstance(answer, dict)
            and answer.get("answer_mode") == "markdown"
            and answer.get("answer_type") in GISA_MARKDOWN_ANSWER_TYPES
        ):
            continue

        samples = raw_result.get("samples", [])
        if not samples:
            continue

        sample_scores = [
            evaluate_gisa_markdown_scores(sample.get("final_answer"), answer)[0]
            for sample in samples
        ]
        exact_match_groups.append(
            [scores.get("exact_match", 0.0) for scores in sample_scores]
        )
        if answer.get("answer_type") == "table":
            row_f1_groups.append(
                [scores.get("row_f1", 0.0) for scores in sample_scores]
            )
        elif answer.get("answer_type") == "list":
            order_score_groups.append(
                [scores.get("order_score", 0.0) for scores in sample_scores]
            )

    metrics = _aggregate_grouped_scores(exact_match_groups, "exact_match")
    metrics.update(_aggregate_grouped_scores(row_f1_groups, "row_f1"))
    metrics.update(_aggregate_grouped_scores(order_score_groups, "order_score"))
    return metrics
