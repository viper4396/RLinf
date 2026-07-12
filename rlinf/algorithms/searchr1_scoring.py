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

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize an answer for Search-R1 exact-match scoring."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punctuation(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(character for character in value if character not in exclude)

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))


def em_check(prediction: str, golden_answers: str | list[str]) -> int:
    """Return one when a prediction exactly matches any normalized answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    return int(
        any(
            normalize_answer(golden_answer) == normalized_prediction
            for golden_answer in golden_answers
        )
    )


def subem_check(prediction: str, golden_answers: str | list[str]) -> int:
    """Return one when any normalized answer is a prediction substring."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    return int(
        any(
            normalize_answer(golden_answer) in normalized_prediction
            for golden_answer in golden_answers
        )
    )


def extract_solution(solution_str: str) -> str | None:
    """Extract the final answer-tag payload from a generated response."""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def count_answer_tags(text: str) -> tuple[int, int]:
    """Count opening and closing answer tags."""
    return text.count("<answer>"), text.count("</answer>")


def compute_score(
    solution_str: str,
    ground_truth: str | list[str],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    do_print: bool = True,
) -> float:
    """Compute Search-R1 normalized exact-match reward."""
    del method
    answer = extract_solution(solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0.0
    if not em_check(answer, ground_truth):
        return format_score
    if open_count > 10 or close_count > 10:
        return score / 4
    return score


def compute_score_subem(
    solution_str: str,
    ground_truth: dict,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    do_print: bool = True,
) -> float:
    """Compute Search-R1 normalized substring exact-match reward."""
    del method
    answer = extract_solution(solution_str)
    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0.0
    if not subem_check(answer, ground_truth["target"]):
        return format_score
    return score
