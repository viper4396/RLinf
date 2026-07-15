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

import datetime
import json
import logging
import os
import typing
from collections import defaultdict
from typing import Optional, Union

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from rlinf.agents.searchr1.reference_runner import SearchR1ReferenceRunnerMixin
from rlinf.agents.searchr1.teacher_planner import build_shadow_metrics
from rlinf.algorithms.searchr1_scoring import subem_check
from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.runners.agent_eval_runner import AgentEvalRunner
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import local_mkdir_safe
from rlinf.workers.agent.agent_loop import MultiAgentLoopWorker
from rlinf.workers.agent.tool_worker import ToolWorker, ToolWorkerInfo
from rlinf.workers.reward.reward_worker import RewardWorker

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class Searchr1AgentEvalRunner(SearchR1ReferenceRunnerMixin, AgentEvalRunner):
    """Runner for Search-R1 evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
        reward: Optional[RewardWorker],
        agent_loop: MultiAgentLoopWorker,
        tool_workers: dict[ToolWorker, ToolWorkerInfo] = {},
        solid_rollouts: dict[str, Union["SGLangWorker", "VLLMWorker"]] = {},
    ):
        super().__init__(
            cfg,
            placement,
            val_dataset,
            rollout,
            reward,
            agent_loop,
            tool_workers,
            solid_rollouts,
        )
        self._init_searchr1_reference_channel()
        # Initialize storage for accumulating evaluation results across all batches
        self.accumulated_results = []

    def _save_eval_results(
        self, all_results, accuracy, total_count, question_count, metrics
    ):
        """Save evaluation results to JSON file.

        Args:
            all_results: List of result dictionaries for each sample
            accuracy: Overall accuracy score
            total_count: Total number of samples evaluated
        """
        # Create output directory in the experiment folder
        output_dir = os.path.join(
            self.cfg.runner.output_dir, self.cfg.runner.experiment_name
        )
        local_mkdir_safe(output_dir)

        # Fixed filename (no timestamp)
        output_file = os.path.join(output_dir, "eval_results.json")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare complete results structure
        results_data = {
            "summary": {
                "dataset_size": total_count,
                "question_count": question_count,
                "correct_count": sum(1 for r in all_results if r["is_correct"]),
                "accuracy": accuracy,
                "experiment_name": self.cfg.runner.experiment_name,
                "timestamp": timestamp,
                "metrics": metrics,
                "config": {
                    "data_paths": OmegaConf.to_container(
                        self.cfg.data.val_data_paths, resolve=True
                    ),
                },
            },
            "results": all_results,
        }

        # Write results to JSON with readable formatting
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Evaluation results saved to: {output_file}")
        return output_file

    def update(
        self,
        context: dict,
        eval_pbar,
        input_channel,
        batch_idx,
        batch,
    ):
        """Collect evaluation results and compute metrics for a single batch.

        This function:
        1. Collects rollout results from the reward channel for one batch
        2. Computes batch accuracy metrics
        3. Accumulates results (does NOT save to file yet)
        """
        recv_batch_size = 0
        group_size = self.cfg.algorithm.get("group_size", 1)
        assert group_size == 1, f"searchr1 eval requires group_size=1, got {group_size}"

        correct_count = 0
        total_count = 0

        while recv_batch_size < self.total_batch_size:
            rollout_result: DynamicRolloutResult = input_channel.get()
            eval_pbar.update(group_size)
            recv_batch_size += group_size

            context["total_questions"] += 1
            extra_fields_traj = rollout_result.extra_fields_traj or {}
            extra_fields_group = rollout_result.extra_fields_group or {}

            answer = extra_fields_group.get("answer", None)
            llm_rewards = extra_fields_traj.get("llm_reward", [0.0])
            response_texts = extra_fields_traj.get("response_text", [None])
            prompt_texts = extra_fields_traj.get("prompt_text", [None])
            turns_list = extra_fields_traj.get("turns", [[]])
            guidance_modes = extra_fields_traj.get(
                "guidance_mode", ["unguided"] * rollout_result.group_size
            )
            if len(guidance_modes) != rollout_result.group_size:
                raise ValueError(
                    "Search-R1 guidance_mode must align with trajectory group"
                )

            metadata_keys = (
                "sample_id",
                "trajectory_id",
                "conditioning_group_id",
                "teacher_version",
                "teacher_plan_id",
                "teacher_plan_node_id",
                "teacher_plan_valid",
                "teacher_plan",
                "teacher_plan_error",
                "teacher_cache_hit",
                "guidance_applied",
                "policy_version",
            )
            group_mode_rewards: dict[str, list[float]] = defaultdict(list)
            group_mode_queries: dict[str, list[str | None]] = defaultdict(list)
            group_mode_answer_hits: dict[str, list[float]] = defaultdict(list)

            for trajectory_idx in range(rollout_result.group_size):
                reward = llm_rewards[trajectory_idx]
                if hasattr(reward, "item"):
                    reward = reward.item()
                reward = float(reward)
                mode = str(guidance_modes[trajectory_idx])
                turns = turns_list[trajectory_idx]
                first_query = next(
                    (
                        turn.get("search_query")
                        for turn in turns
                        if turn.get("is_search")
                    ),
                    None,
                )
                visible_evidence = "\n".join(
                    str(turn.get("visible_evidence") or "") for turn in turns
                )
                answer_hit = (
                    float(subem_check(visible_evidence, answer))
                    if answer is not None and visible_evidence
                    else 0.0
                )

                group_mode_rewards[mode].append(reward)
                group_mode_queries[mode].append(first_query)
                group_mode_answer_hits[mode].append(answer_hit)
                context["mode_reward_sums"][mode] += reward
                context["mode_counts"][mode] += 1
                context["mode_answer_hit_sums"][mode] += answer_hit

                is_correct = reward > 0
                correct_count += int(is_correct)
                total_count += 1
                result_entry = {
                    "index": len(self.accumulated_results),
                    "trajectory_index": trajectory_idx,
                    "prompt_text": prompt_texts[trajectory_idx],
                    "turns": turns,
                    "response_text": response_texts[trajectory_idx],
                    "answer": answer,
                    "guidance_mode": mode,
                    "reward": reward,
                    "is_correct": is_correct,
                    "answer_hit": bool(answer_hit),
                }
                for key in metadata_keys:
                    values = extra_fields_traj.get(key)
                    result_entry[key] = (
                        values[trajectory_idx] if values is not None else None
                    )
                self.accumulated_results.append(result_entry)

                plan_id = result_entry["teacher_plan_id"]
                if plan_id is not None:
                    context["plan_valid_by_id"][plan_id] = bool(
                        result_entry["teacher_plan_valid"]
                    )
                    context["plan_cache_hit_by_id"][plan_id] = bool(
                        result_entry["teacher_cache_hit"]
                    )

            if "unguided" in group_mode_rewards:
                unguided_mean = sum(group_mode_rewards["unguided"]) / len(
                    group_mode_rewards["unguided"]
                )
                unguided_hit_mean = sum(group_mode_answer_hits["unguided"]) / len(
                    group_mode_answer_hits["unguided"]
                )
                for mode, mode_rewards in group_mode_rewards.items():
                    if mode == "unguided":
                        continue
                    mode_mean = sum(mode_rewards) / len(mode_rewards)
                    mode_hit_mean = sum(group_mode_answer_hits[mode]) / len(
                        group_mode_answer_hits[mode]
                    )
                    context["paired_uplift_sums"][mode] += mode_mean - unguided_mean
                    context["paired_uplifts"][mode].append(mode_mean - unguided_mean)
                    context["paired_answer_hit_sums"][mode] += (
                        mode_hit_mean - unguided_hit_mean
                    )
                    context["paired_counts"][mode] += 1

                    mode_queries = group_mode_queries[mode]
                    unguided_queries = group_mode_queries["unguided"]
                    query_pairs = zip(mode_queries, unguided_queries)
                    for mode_query, unguided_query in query_pairs:
                        context["query_change_sums"][mode] += float(
                            mode_query != unguided_query
                        )
                        context["query_change_counts"][mode] += 1

        # Compute batch accuracy
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        # Log batch statistics
        logging.info("Batch Evaluation Summary:")
        logging.info(f"  Batch samples: {total_count}")
        logging.info(f"  Batch correct: {correct_count}")
        logging.info(f"  Batch accuracy: {accuracy:.4f}")
        logging.info(f"  Total accumulated samples: {len(self.accumulated_results)}")

        batch_correct = int(accuracy * total_count)
        context["total_correct"] += batch_correct
        context["total_samples"] += total_count
        context["batch_accuracy"] = accuracy

    def pre_process(self) -> dict:
        logging.info("=" * 80)
        logging.info("Starting Search-R1 Evaluation")
        logging.info("=" * 80)
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")
        logging.info(f"Batch size: {self.val_batch_size}")
        logging.info(f"Group size: {self.cfg.algorithm.get('group_size', 1)}")
        logging.info(f"Max turns: {self.cfg.agentloop.get('max_turns', 5)}")
        logging.info("=" * 80)

        context = {
            "total_correct": 0,
            "total_samples": 0,
            "total_questions": 0,
            "mode_reward_sums": defaultdict(float),
            "mode_answer_hit_sums": defaultdict(float),
            "mode_counts": defaultdict(int),
            "paired_uplift_sums": defaultdict(float),
            "paired_uplifts": defaultdict(list),
            "paired_answer_hit_sums": defaultdict(float),
            "paired_counts": defaultdict(int),
            "query_change_sums": defaultdict(float),
            "query_change_counts": defaultdict(int),
            "plan_valid_by_id": {},
            "plan_cache_hit_by_id": {},
        }
        return context

    def post_process(
        self,
        context: dict,
    ) -> dict:
        total_correct = context["total_correct"]
        total_samples = context["total_samples"]
        question_count = context["total_questions"]
        teacher_cfg = self.cfg.get("teacher_planner", {})
        shadow_metrics = build_shadow_metrics(
            context,
            bootstrap_seed=int(teacher_cfg.get("seed", self.cfg.data.get("seed", 0))),
            bootstrap_samples=int(teacher_cfg.get("bootstrap_samples", 2000)),
        )
        # Final summary
        final_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        logging.info("\n" + "=" * 80)
        logging.info("EVALUATION COMPLETED")
        logging.info("=" * 80)
        logging.info(f"Total samples evaluated: {total_samples}")
        logging.info(f"Total correct: {total_correct}")
        logging.info(
            f"Final accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)"
        )
        for metric_name, metric_value in sorted(shadow_metrics.items()):
            logging.info(f"{metric_name}: {metric_value:.4f}")
        logging.info("=" * 80)

        self.metric_logger.log(shadow_metrics, step=self.global_steps)

        # Save all accumulated results to JSON file
        logging.info(f"Saving {len(self.accumulated_results)} results to JSON file...")
        self._save_eval_results(
            self.accumulated_results,
            final_accuracy,
            total_samples,
            question_count,
            shadow_metrics,
        )

    def update_batch(
        self,
        context: dict,
        eval_pbar,
        time_metrics,
    ):
        # Update progress bar with current metrics
        total_correct = context["total_correct"]
        total_samples = context["total_samples"]
        batch_accuracy = context["batch_accuracy"]
        current_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        eval_pbar.set_postfix(
            {
                "batch_acc": f"{batch_accuracy:.4f}",
                "overall_acc": f"{current_accuracy:.4f}",
                "samples": total_samples,
                "rollout_time": f"{time_metrics.get('rollout', 0):.2f}s",
            }
        )
