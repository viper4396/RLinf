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

import random

from omegaconf import DictConfig

from rlinf.algorithms.searchr1_scoring import (
    compute_score,
    compute_score_subem,
    count_answer_tags,
    em_check,
    extract_solution,
    normalize_answer,
    subem_check,
)

__all__ = [
    "SearchR1Reward",
    "compute_score",
    "compute_score_subem",
    "count_answer_tags",
    "em_check",
    "extract_solution",
    "normalize_answer",
    "subem_check",
]


class SearchR1Reward:
    def __init__(self, config: DictConfig):
        self.scale = config.get("reward_scale", 1.0)
        self.random_print_percent = config.get("random_print_percent", 0.01)

    def get_reward(
        self, response: list[str], reference: list[list[str]]
    ) -> list[float]:
        if self.random_print_percent <= 0:
            do_prints = [False for _ in range(len(response))]
        elif self.random_print_percent >= 1:
            do_prints = [True for _ in range(len(response))]
        else:
            do_prints = [
                random.random() < self.random_print_percent
                for _ in range(len(response))
            ]
        rewards = [
            compute_score(sol, gt, do_print=do_print)
            for sol, gt, do_print in zip(response, reference, do_prints)
        ]
        return rewards
