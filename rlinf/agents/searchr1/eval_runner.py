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
import hashlib
import json
import logging
import os
import re
import typing
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from rlinf.agents.searchr1.eval_diagnostics import (
    build_abc_acceptance_metrics,
    build_label_only_diagnostics,
)
from rlinf.agents.searchr1.reference_runner import SearchR1ReferenceRunnerMixin
from rlinf.agents.searchr1.teacher_planner import build_shadow_metrics
from rlinf.algorithms.searchr1_scoring import extract_solution, subem_check
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


def build_searchr1_gisa_metrics(context: dict[str, Any]) -> dict[str, float]:
    """Aggregate Search-R1 GISA structural metrics from evaluation state."""
    gisa_count = int(context.get("gisa_count", 0))
    if not gisa_count:
        return {}

    metrics = {
        "gisa/cell_f1": context["gisa_cell_f1_sum"] / gisa_count,
        "gisa/exact_match": context["gisa_exact_match_sum"] / gisa_count,
        "gisa/format_rate": context["gisa_format_sum"] / gisa_count,
    }
    metrics["gisa/pass@1"] = metrics["gisa/exact_match"]
    if context["gisa_row_f1_count"]:
        metrics["gisa/table_row_f1"] = (
            context["gisa_row_f1_sum"] / context["gisa_row_f1_count"]
        )
    if context["gisa_order_score_count"]:
        metrics["gisa/list_order_score"] = (
            context["gisa_order_score_sum"] / context["gisa_order_score_count"]
        )
    for answer_type, type_count in context["gisa_type_counts"].items():
        prefix = f"gisa/type/{answer_type}"
        metrics[f"{prefix}/cell_f1"] = (
            context["gisa_type_cell_f1_sums"][answer_type] / type_count
        )
        metrics[f"{prefix}/exact_match"] = (
            context["gisa_type_exact_match_sums"][answer_type] / type_count
        )
        metrics[f"{prefix}/format_rate"] = (
            context["gisa_type_format_sums"][answer_type] / type_count
        )
    return metrics


def _sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using one canonical representation."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _content_tree_hash(paths: list[str]) -> str:
    """Hash file names and bytes for small, evaluation-critical artifacts."""
    digest = hashlib.sha256()
    for raw_path in sorted(set(paths)):
        path = Path(raw_path).expanduser()
        candidates = (
            sorted(item for item in path.rglob("*") if item.is_file())
            if path.is_dir()
            else [path]
        )
        for candidate in candidates:
            relative_name = (
                str(candidate.relative_to(path)) if path.is_dir() else candidate.name
            )
            digest.update(str(path).encode("utf-8"))
            digest.update(relative_name.encode("utf-8"))
            if not candidate.is_file():
                digest.update(b"<missing>")
                continue
            with candidate.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _model_manifest_hash(model_path: str) -> str:
    """Hash checkpoint layout plus small model/tokenizer metadata files."""
    root = Path(model_path).expanduser()
    if not root.is_dir():
        return _sha256_json({"model_path": str(root), "status": "missing"})
    entries: list[dict[str, Any]] = []
    metadata_suffixes = {".json", ".txt", ".model", ".py", ".tiktoken"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entry: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
        }
        if path.suffix.casefold() in metadata_suffixes:
            entry["content_sha256"] = _content_tree_hash([str(path)])
        entries.append(entry)
    return _sha256_json(entries)


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

    def _validate_complete_shadow_results(self) -> None:
        """Reject incomplete, duplicated, or unbalanced paired A/B output."""
        teacher_cfg = self.cfg.get("teacher_planner", {})
        if not teacher_cfg.get("enabled", False):
            return
        expected_modes = sorted(str(mode) for mode in teacher_cfg.guidance_modes)
        expected_question_count = len(self.val_dataset)
        expected_result_count = expected_question_count * len(expected_modes)
        if len(self.accumulated_results) != expected_result_count:
            raise RuntimeError(
                "incomplete Search-R1 shadow output: expected "
                f"{expected_result_count}, got {len(self.accumulated_results)}"
            )

        by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        trajectory_ids: list[str] = []
        for result in self.accumulated_results:
            by_sample[str(result.get("sample_id"))].append(result)
            trajectory_id = str(result.get("trajectory_id") or "")
            if not trajectory_id:
                raise RuntimeError(
                    "Search-R1 shadow output has a missing trajectory ID"
                )
            trajectory_ids.append(trajectory_id)
        if len(by_sample) != expected_question_count:
            raise RuntimeError(
                "Search-R1 shadow output contains missing or duplicate sample IDs: "
                f"expected {expected_question_count}, got {len(by_sample)}"
            )
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise RuntimeError(
                "Search-R1 shadow output contains duplicate trajectories"
            )
        for sample_id, sample_results in by_sample.items():
            actual_modes = sorted(
                str(result.get("guidance_mode")) for result in sample_results
            )
            if actual_modes != expected_modes:
                raise RuntimeError(
                    f"unbalanced A/B modes for sample {sample_id}: "
                    f"expected {expected_modes}, got {actual_modes}"
                )

    def _reproducibility_manifest(self) -> dict[str, Any]:
        """Build immutable identifiers for every A/B-critical input."""
        resolved_cfg = OmegaConf.to_container(self.cfg, resolve=True)
        data_paths = [str(path) for path in self.cfg.data.val_data_paths]
        teacher_cfg = self.cfg.get("teacher_planner", {})
        cache_dir = str(teacher_cfg.get("cache_dir") or "")
        policy_model_path = str(self.cfg.rollout.model.model_path)
        teacher_model_path = str(teacher_cfg.get("model", {}).get("model_path") or "")
        source_root = Path(__file__).parent
        controller_sources = [
            str(source_root / "eval_runner.py"),
            str(source_root / "searchr1_agent_loop.py"),
            str(source_root / "teacher_planner.py"),
            str(source_root / "eval_diagnostics.py"),
        ]
        retrieval_config = OmegaConf.to_container(self.cfg.tools.search, resolve=True)
        return {
            "resolved_config_sha256": _sha256_json(resolved_cfg),
            "dataset_sha256": _content_tree_hash(data_paths),
            "data_paths": data_paths,
            "policy_version": str(
                self.cfg.agentloop.get("policy_version", policy_model_path)
            ),
            "policy_model_path": policy_model_path,
            "policy_model_manifest_sha256": _model_manifest_hash(policy_model_path),
            "teacher_version": str(teacher_cfg.get("version") or ""),
            "teacher_model_path": teacher_model_path,
            "teacher_model_manifest_sha256": (
                _model_manifest_hash(teacher_model_path) if teacher_model_path else None
            ),
            "teacher_plan_cache_dir": cache_dir,
            "teacher_plan_cache_sha256": (
                _content_tree_hash([cache_dir]) if cache_dir else None
            ),
            "controller_source_sha256": _content_tree_hash(controller_sources),
            "retrieval_config": retrieval_config,
            "retrieval_config_sha256": _sha256_json(retrieval_config),
            "data_seed": int(self.cfg.data.get("seed", 0)),
            "teacher_seed": int(teacher_cfg.get("seed", 0)),
            "policy_sampling_params": OmegaConf.to_container(
                self.cfg.algorithm.sampling_params, resolve=True
            ),
        }

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
                "reproducibility": self._reproducibility_manifest(),
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

    def _save_partial_results(self, next_batch_idx: int) -> str:
        """Atomically checkpoint completed batches for failure recovery."""
        output_dir = os.path.join(
            self.cfg.runner.output_dir, self.cfg.runner.experiment_name
        )
        local_mkdir_safe(output_dir)
        output_file = os.path.join(output_dir, "eval_results.partial.json")
        temporary_file = f"{output_file}.tmp"
        payload = {
            "summary": {
                "dataset_size": len(self.val_dataset),
                "completed_count": len(self.accumulated_results),
                "next_batch_idx": next_batch_idx,
                "experiment_name": self.cfg.runner.experiment_name,
                "data_paths": OmegaConf.to_container(
                    self.cfg.data.val_data_paths, resolve=True
                ),
            },
            "results": self.accumulated_results,
        }
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary_file, output_file)
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
        expected_batch_size = len(batch["answer"]) * group_size

        correct_count = 0
        total_count = 0

        while recv_batch_size < expected_batch_size:
            rollout_result: DynamicRolloutResult = input_channel.get()
            eval_pbar.update(group_size)
            recv_batch_size += group_size

            context["total_questions"] += 1
            extra_fields_traj = rollout_result.extra_fields_traj or {}
            extra_fields_group = rollout_result.extra_fields_group or {}

            answer = extra_fields_group.get("answer", None)
            llm_rewards = extra_fields_traj.get("llm_reward", [0.0])
            em_rewards = extra_fields_traj.get("em_reward", llm_rewards)
            judge_rewards = extra_fields_traj.get("judge_reward")
            judge_responses = extra_fields_traj.get("judge_response")
            gisa_metrics_values = extra_fields_traj.get("gisa_metrics")
            gisa_format_values = extra_fields_traj.get("gisa_format_ok")
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
                "teacher_decision",
                "teacher_plan_type",
                "teacher_plan_step_count",
                "teacher_execution_mode",
                "teacher_controller_applied",
                "teacher_rewrite_applied",
                "guidance_applied",
                "controller_completed_step_ids",
                "controller_template_query_count",
                "controller_policy_query_count",
                "controller_fallback_query_count",
                "controller_dependent_query_count",
                "controller_binding_valid_count",
                "controller_binding_attempt_count",
                "controller_binding_alias_count",
                "controller_binding_failure_reasons",
                "controller_resolved_values_by_step",
                "controller_synthesis_generated",
                "controller_synthesis_answer_source",
                "controller_synthesis_format_repaired",
                "controller_synthesis_format_valid",
                "controller_completed",
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
                em_reward = float(em_rewards[trajectory_idx])
                judge_reward = (
                    judge_rewards[trajectory_idx] if judge_rewards is not None else None
                )
                mode = str(guidance_modes[trajectory_idx])
                turns = turns_list[trajectory_idx]
                is_gisa = isinstance(answer, dict) and bool(
                    answer.get("is_gisa", False)
                )
                gisa_metrics = (
                    gisa_metrics_values[trajectory_idx]
                    if gisa_metrics_values is not None
                    else {}
                )
                gisa_format_ok = bool(
                    gisa_format_values is not None
                    and gisa_format_values[trajectory_idx]
                )
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
                    0.0
                    if is_gisa
                    else (
                        float(subem_check(visible_evidence, answer))
                        if answer is not None and visible_evidence
                        else 0.0
                    )
                )
                final_answer = extract_solution(response_texts[trajectory_idx] or "")
                diagnostic_subem = (
                    0.0
                    if is_gisa
                    else (
                        float(subem_check(final_answer, answer))
                        if answer is not None and final_answer is not None
                        else 0.0
                    )
                )
                first_search_turn = next(
                    (turn for turn in turns if turn.get("is_search")), None
                )
                tool_call_repaired = float(
                    bool(
                        first_search_turn
                        and first_search_turn.get("tool_call_repaired", False)
                    )
                )
                dual_query_applied = float(
                    bool(
                        first_search_turn
                        and first_search_turn.get("dual_query_applied", False)
                    )
                )

                group_mode_rewards[mode].append(reward)
                group_mode_queries[mode].append(first_query)
                group_mode_answer_hits[mode].append(answer_hit)
                context["mode_reward_sums"][mode] += reward
                context["mode_counts"][mode] += 1
                context["mode_answer_hit_sums"][mode] += answer_hit
                context["mode_subem_sums"][mode] += diagnostic_subem
                context["mode_tool_call_repair_sums"][mode] += tool_call_repaired
                context["mode_dual_query_sums"][mode] += dual_query_applied
                controller_completed_values = extra_fields_traj.get(
                    "controller_completed"
                )
                controller_applied_values = extra_fields_traj.get(
                    "teacher_controller_applied"
                )
                controller_fallback_values = extra_fields_traj.get(
                    "controller_fallback_query_count"
                )
                controller_step_values = extra_fields_traj.get(
                    "controller_completed_step_ids"
                )
                controller_dependent_values = extra_fields_traj.get(
                    "controller_dependent_query_count"
                )
                controller_binding_valid_values = extra_fields_traj.get(
                    "controller_binding_valid_count"
                )
                controller_binding_attempt_values = extra_fields_traj.get(
                    "controller_binding_attempt_count"
                )
                controller_binding_alias_values = extra_fields_traj.get(
                    "controller_binding_alias_count"
                )
                synthesis_repaired_values = extra_fields_traj.get(
                    "controller_synthesis_format_repaired"
                )
                synthesis_valid_values = extra_fields_traj.get(
                    "controller_synthesis_format_valid"
                )
                context["mode_controller_completion_sums"][mode] += float(
                    bool(
                        controller_completed_values
                        and controller_completed_values[trajectory_idx]
                    )
                )
                context["mode_controller_applied_sums"][mode] += float(
                    bool(
                        controller_applied_values
                        and controller_applied_values[trajectory_idx]
                    )
                )
                context["mode_controller_fallback_query_sums"][mode] += float(
                    controller_fallback_values[trajectory_idx]
                    if controller_fallback_values
                    else 0
                )
                context["mode_controller_step_sums"][mode] += float(
                    len(controller_step_values[trajectory_idx])
                    if controller_step_values
                    else 0
                )
                context["mode_controller_dependent_step_sums"][mode] += float(
                    controller_dependent_values[trajectory_idx]
                    if controller_dependent_values
                    else 0
                )
                context["mode_controller_binding_valid_sums"][mode] += float(
                    controller_binding_valid_values[trajectory_idx]
                    if controller_binding_valid_values
                    else 0
                )
                context["mode_controller_binding_attempt_sums"][mode] += float(
                    controller_binding_attempt_values[trajectory_idx]
                    if controller_binding_attempt_values
                    else 0
                )
                context["mode_controller_binding_alias_sums"][mode] += float(
                    controller_binding_alias_values[trajectory_idx]
                    if controller_binding_alias_values
                    else 0
                )
                context["mode_synthesis_format_repair_sums"][mode] += float(
                    bool(
                        synthesis_repaired_values
                        and synthesis_repaired_values[trajectory_idx]
                    )
                )
                context["mode_synthesis_format_valid_sums"][mode] += float(
                    bool(
                        synthesis_valid_values
                        and synthesis_valid_values[trajectory_idx]
                    )
                )
                turn_count = len(turns)
                search_count = sum(
                    int(bool(turn.get("is_search", False))) for turn in turns
                )
                generated_token_count = sum(
                    int(turn.get("generated_token_count", 0) or 0) for turn in turns
                )
                context["mode_turn_sums"][mode] += turn_count
                context["mode_search_sums"][mode] += search_count
                context["mode_generated_token_sums"][mode] += generated_token_count
                failure_reason_values = extra_fields_traj.get(
                    "controller_binding_failure_reasons"
                )
                failure_reasons = (
                    failure_reason_values[trajectory_idx]
                    if failure_reason_values
                    else {}
                )
                for reason, reason_count in (failure_reasons or {}).items():
                    context["mode_binding_failure_reason_sums"][mode][reason] += int(
                        reason_count
                    )
                unresolved_placeholders = sum(
                    int(
                        bool(
                            re.search(
                                r"\{?step[_ ]?\d+(?:[_ ]result)?\}?",
                                str(query),
                                re.IGNORECASE,
                            )
                        )
                    )
                    for turn in turns
                    for query in (turn.get("executed_search_queries") or [])
                )
                context["mode_unresolved_placeholder_sums"][mode] += (
                    unresolved_placeholders
                )

                if is_gisa:
                    answer_type = str(answer.get("answer_type", "table"))
                    gisa_exact_match = float(
                        gisa_metrics.get(
                            "pass",
                            gisa_metrics.get("exact_match", 0.0),
                        )
                    )
                    gisa_cell_f1 = float(gisa_metrics.get("cell_f1", reward))
                    context["gisa_count"] += 1
                    context["gisa_cell_f1_sum"] += gisa_cell_f1
                    context["gisa_exact_match_sum"] += gisa_exact_match
                    context["gisa_format_sum"] += float(gisa_format_ok)
                    context["gisa_type_counts"][answer_type] += 1
                    context["gisa_type_cell_f1_sums"][answer_type] += gisa_cell_f1
                    context["gisa_type_exact_match_sums"][answer_type] += (
                        gisa_exact_match
                    )
                    context["gisa_type_format_sums"][answer_type] += float(
                        gisa_format_ok
                    )
                    if "row_f1" in gisa_metrics:
                        context["gisa_row_f1_sum"] += float(gisa_metrics["row_f1"])
                        context["gisa_row_f1_count"] += 1
                    if "order_score" in gisa_metrics:
                        context["gisa_order_score_sum"] += float(
                            gisa_metrics["order_score"]
                        )
                        context["gisa_order_score_count"] += 1
                    is_correct = gisa_exact_match == 1.0
                else:
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
                    "em_reward": em_reward,
                    "judge_reward": judge_reward,
                    "judge_response": (
                        judge_responses[trajectory_idx]
                        if judge_responses is not None
                        else None
                    ),
                    "gisa_metrics": gisa_metrics if is_gisa else None,
                    "gisa_format_ok": gisa_format_ok if is_gisa else None,
                    "is_correct": is_correct,
                    "answer_hit": bool(answer_hit),
                    "diagnostic_subem": bool(diagnostic_subem),
                    "turn_count": turn_count,
                    "search_count": search_count,
                    "generated_token_count": generated_token_count,
                }
                for key in metadata_keys:
                    values = extra_fields_traj.get(key)
                    result_entry[key] = (
                        values[trajectory_idx] if values is not None else None
                    )
                self.accumulated_results.append(result_entry)
                context["em_reward_sum"] += em_reward
                if judge_reward is not None:
                    context["judge_reward_sum"] += float(judge_reward)
                    context["judge_reward_count"] += 1

                plan_id = result_entry["teacher_plan_id"]
                if plan_id is not None:
                    context["plan_valid_by_id"][plan_id] = bool(
                        result_entry["teacher_plan_valid"]
                    )
                    context["plan_cache_hit_by_id"][plan_id] = bool(
                        result_entry["teacher_cache_hit"]
                    )
                    context["plan_decision_by_id"][plan_id] = result_entry[
                        "teacher_decision"
                    ]

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

        context["total_correct"] += correct_count
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

        teacher_cfg = self.cfg.get("teacher_planner", {})
        if teacher_cfg.get("enabled", False):
            index_manifest = self.cfg.tools.search.get("index_manifest", {})
            index_hash = str(index_manifest.get("manifest_sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", index_hash):
                raise ValueError(
                    "teacher shadow evaluation requires tools.search.index_manifest."
                    "manifest_sha256"
                )

        context = {
            "total_correct": 0,
            "total_samples": 0,
            "total_questions": 0,
            "em_reward_sum": 0.0,
            "judge_reward_sum": 0.0,
            "judge_reward_count": 0,
            "gisa_count": 0,
            "gisa_cell_f1_sum": 0.0,
            "gisa_exact_match_sum": 0.0,
            "gisa_format_sum": 0.0,
            "gisa_row_f1_sum": 0.0,
            "gisa_row_f1_count": 0,
            "gisa_order_score_sum": 0.0,
            "gisa_order_score_count": 0,
            "gisa_type_counts": defaultdict(int),
            "gisa_type_cell_f1_sums": defaultdict(float),
            "gisa_type_exact_match_sums": defaultdict(float),
            "gisa_type_format_sums": defaultdict(float),
            "mode_reward_sums": defaultdict(float),
            "mode_answer_hit_sums": defaultdict(float),
            "mode_subem_sums": defaultdict(float),
            "mode_tool_call_repair_sums": defaultdict(float),
            "mode_dual_query_sums": defaultdict(float),
            "mode_controller_completion_sums": defaultdict(float),
            "mode_controller_applied_sums": defaultdict(float),
            "mode_controller_fallback_query_sums": defaultdict(float),
            "mode_controller_step_sums": defaultdict(float),
            "mode_controller_dependent_step_sums": defaultdict(float),
            "mode_controller_binding_valid_sums": defaultdict(float),
            "mode_controller_binding_attempt_sums": defaultdict(float),
            "mode_controller_binding_alias_sums": defaultdict(float),
            "mode_synthesis_format_repair_sums": defaultdict(float),
            "mode_synthesis_format_valid_sums": defaultdict(float),
            "mode_turn_sums": defaultdict(float),
            "mode_search_sums": defaultdict(float),
            "mode_generated_token_sums": defaultdict(float),
            "mode_binding_failure_reason_sums": defaultdict(lambda: defaultdict(float)),
            "mode_unresolved_placeholder_sums": defaultdict(float),
            "time_metric_sums": defaultdict(float),
            "mode_counts": defaultdict(int),
            "paired_uplift_sums": defaultdict(float),
            "paired_uplifts": defaultdict(list),
            "paired_answer_hit_sums": defaultdict(float),
            "paired_counts": defaultdict(int),
            "query_change_sums": defaultdict(float),
            "query_change_counts": defaultdict(int),
            "plan_valid_by_id": {},
            "plan_cache_hit_by_id": {},
            "plan_decision_by_id": {},
        }
        return context

    def post_process(
        self,
        context: dict,
    ) -> dict:
        total_correct = context["total_correct"]
        total_samples = context["total_samples"]
        question_count = context["total_questions"]
        self._validate_complete_shadow_results()
        teacher_cfg = self.cfg.get("teacher_planner", {})
        shadow_metrics = build_shadow_metrics(
            context,
            bootstrap_seed=int(teacher_cfg.get("seed", self.cfg.data.get("seed", 0))),
            bootstrap_samples=int(teacher_cfg.get("bootstrap_samples", 2000)),
        )
        dataset_records = getattr(self.val_dataset, "data", [])
        if isinstance(dataset_records, list):
            shadow_metrics.update(
                build_label_only_diagnostics(
                    self.accumulated_results,
                    dataset_records,
                    bootstrap_seed=int(
                        teacher_cfg.get("seed", self.cfg.data.get("seed", 0))
                    ),
                    bootstrap_samples=int(teacher_cfg.get("bootstrap_samples", 2000)),
                )
            )
        for metric_name, metric_value in context["time_metric_sums"].items():
            shadow_metrics[f"time/{metric_name}_seconds"] = float(metric_value)
        shadow_metrics.update(build_abc_acceptance_metrics(shadow_metrics))
        shadow_metrics["eval/exact_match"] = (
            context["em_reward_sum"] / total_samples if total_samples else 0.0
        )
        if context["judge_reward_count"]:
            shadow_metrics["eval/judge_accuracy"] = (
                context["judge_reward_sum"] / context["judge_reward_count"]
            )
        shadow_metrics.update(build_searchr1_gisa_metrics(context))
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
        for metric_name, metric_value in time_metrics.items():
            context["time_metric_sums"][metric_name] += float(metric_value)
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
        self._save_partial_results(self.global_steps + 1)
