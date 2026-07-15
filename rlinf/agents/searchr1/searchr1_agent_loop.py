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
import hashlib
from typing import Any
from uuid import uuid4

from omegaconf import DictConfig
from transformers import AutoTokenizer

from rlinf.agents.searchr1.teacher_planner import (
    FrozenTeacherPlanner,
    TeacherPlanResult,
    build_guidance_token_ids,
    shuffled_teacher_plans,
)
from rlinf.data.io_struct import RolloutRequest
from rlinf.data.tool_call.tool_io_struct import ToolResponse
from rlinf.scheduler import Channel
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)


def truncate_token_ids(
    token_ids: list[int], max_length: int, truncate_side: str
) -> list[int]:
    """Truncate token IDs while following tokenizer-style side semantics."""
    if max_length < 0:
        raise ValueError("max_length must be non-negative")
    if len(token_ids) <= max_length:
        return token_ids
    if max_length == 0:
        return []
    if truncate_side == "right":
        return token_ids[:max_length]
    if truncate_side == "left":
        return token_ids[-max_length:]
    if truncate_side == "middle":
        left_length = max_length // 2
        right_length = max_length - left_length
        return token_ids[:left_length] + token_ids[-right_length:]
    raise ValueError("tool_response_truncate_side must be one of: left, right, middle")


class Searchr1AgentLoopWorker(MultiAgentLoopWorker):
    """
    Search-R1 agent loop
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        self.max_total_len = int(self.cfg.runner.seq_length)
        self.max_resp_len = max(1, self.max_total_len - self.max_prompt_len)
        self.max_tool_response_length = int(
            self.cfg.agentloop.get("max_tool_response_length", 500)
        )
        self.tool_response_truncate_side = self.cfg.agentloop.get(
            "tool_response_truncate_side", "right"
        )
        if self.max_tool_response_length < 0:
            raise ValueError("max_tool_response_length must be non-negative")
        if self.tool_response_truncate_side not in {"left", "right", "middle"}:
            raise ValueError(
                "tool_response_truncate_side must be one of: left, right, middle"
            )

        assert self.toolcall_parser is not None, (
            "toolcall_parser must be set in searchr1"
        )

        # Inserting tool info requires re-encode token_ids, so the recompute_logprobs must be true.
        if self.cfg.runner.task_type != "reasoning_eval":
            assert self.cfg.algorithm.recompute_logprobs, (
                "search r1 must use recompute_logprobs"
            )

        teacher_cfg = self.cfg.get("teacher_planner", {})
        self.teacher_planner_enabled = bool(teacher_cfg.get("enabled", False))
        self.guidance_modes = list(
            teacher_cfg.get(
                "guidance_modes", ["guided", "guided", "unguided", "unguided"]
            )
        )
        self.teacher_planner = None
        if self.teacher_planner_enabled:
            valid_modes = {"guided", "unguided", "shuffled", "generic"}
            unknown_modes = set(self.guidance_modes) - valid_modes
            if unknown_modes:
                raise ValueError(
                    f"unknown teacher guidance modes: {sorted(unknown_modes)}"
                )
            if len(self.guidance_modes) != int(self.cfg.algorithm.group_size):
                raise ValueError(
                    "teacher_planner.guidance_modes must have algorithm.group_size "
                    "entries"
                )
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_cfg.model.model_path
            )
            self.teacher_planner = FrozenTeacherPlanner(self.cfg, teacher_tokenizer)

    async def pre_process_query(
        self,
        prompt_ids: list[int],
        answer: str,
        *,
        question_text: str | None = None,
        sample_id: str | int | None = None,
        guidance_mode: str = "unguided",
        teacher_plan_result: TeacherPlanResult | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """Prepare a query using an opaque reward-reference ID, never GT."""
        original_prompt_ids = prompt_ids[: self.max_prompt_len]
        guidance_ids: list[int] = []
        if (
            guidance_mode != "unguided"
            and teacher_plan_result is not None
            and teacher_plan_result.valid
            and teacher_plan_result.plan is not None
        ):
            guidance_ids = build_guidance_token_ids(
                self.tokenizer, teacher_plan_result.plan, guidance_mode
            )

        teacher_planner = getattr(self, "teacher_planner", None)
        teacher_version = None
        if teacher_planner is not None:
            teacher_version = teacher_planner.teacher_version
        plan_id = (
            teacher_plan_result.plan_id if teacher_plan_result is not None else None
        )
        if guidance_mode == "unguided":
            conditioning_group_id = f"unguided:{sample_id}"
        elif guidance_mode == "generic":
            conditioning_group_id = f"generic:{plan_id}"
        else:
            conditioning_group_id = plan_id
        cfg = getattr(self, "cfg", {})
        agentloop_cfg = cfg.get("agentloop", {})
        rollout_cfg = cfg.get("rollout", {})
        policy_version = agentloop_cfg.get(
            "policy_version",
            rollout_cfg.get("model", {}).get("model_path", "unknown"),
        )

        return (
            original_prompt_ids + guidance_ids,
            {
                "reference_id": answer,
                "sample_id": sample_id,
                "trajectory_id": uuid4().hex,
                "question_text": question_text,
                "next_turn_id": 0,
                "all_llm_response_ids": [],  # accumulate only LLM-generated tokens for reward
                "problem_prompt_ids": copy.deepcopy(original_prompt_ids),
                "unguided_problem_prompt_ids": copy.deepcopy(original_prompt_ids),
                "last_llm_output": None,
                "guidance_mode": guidance_mode,
                "conditioning_group_id": conditioning_group_id,
                "teacher_version": teacher_version,
                "teacher_plan_id": plan_id,
                "teacher_plan_node_id": f"{plan_id}:first_search"
                if plan_id is not None
                else None,
                "teacher_plan_valid": bool(
                    teacher_plan_result is not None and teacher_plan_result.valid
                ),
                "teacher_plan": (
                    teacher_plan_result.plan.to_dict()
                    if teacher_plan_result is not None
                    and teacher_plan_result.plan is not None
                    else None
                ),
                "teacher_plan_error": (
                    teacher_plan_result.error
                    if teacher_plan_result is not None
                    else None
                ),
                "teacher_cache_hit": bool(
                    teacher_plan_result is not None and teacher_plan_result.cache_hit
                ),
                "guidance_applied": bool(guidance_ids),
                "policy_version": str(policy_version),
            },
        )

    async def post_process_query(
        self, generate_context: dict[str, Any], output: MultiAgentLoopOutput
    ) -> MultiAgentLoopOutput:
        """Finalize text and metadata without accessing a reward reference."""
        if output.single_turn_outputs and not any(
            turn.extra_fields.get("is_terminal", False)
            for turn in output.single_turn_outputs
        ):
            terminal_output = output.single_turn_outputs[-1]
            terminal_output.is_end = True
            terminal_output.extra_fields["is_terminal"] = True

        final_response_text = self.tokenizer.decode(
            generate_context["all_llm_response_ids"]
        )
        for single_turn_output in output.single_turn_outputs:
            single_turn_output.reward_score = 0.0
            metadata = single_turn_output.extra_fields
            metadata["guidance_mode"] = generate_context["guidance_mode"]
            metadata["conditioning_group_id"] = generate_context[
                "conditioning_group_id"
            ]
            metadata["teacher_version"] = generate_context["teacher_version"]
            metadata["trajectory_id"] = generate_context["trajectory_id"]
            metadata["teacher_plan_id"] = generate_context["teacher_plan_id"]
            metadata["teacher_plan_node_id"] = (
                generate_context["teacher_plan_node_id"]
                if metadata["turn_id"] == 0 and generate_context["guidance_applied"]
                else None
            )

        output.extra_fields["llm_reward"] = 0.0
        output.extra_fields["response_text"] = final_response_text
        output.extra_fields["prompt_text"] = self.tokenizer.decode(
            generate_context.get("problem_prompt_ids", [])
        )
        for key in (
            "sample_id",
            "trajectory_id",
            "guidance_mode",
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
        ):
            output.extra_fields[key] = generate_context[key]
        # Per-turn details: each turn's input and output text
        turns = []
        for single_turn_output in output.single_turn_outputs:
            turns.append(
                {
                    "input": self.tokenizer.decode(single_turn_output.prompt_ids),
                    "output": self.tokenizer.decode(single_turn_output.response_ids),
                    "turn_id": single_turn_output.extra_fields["turn_id"],
                    "is_search": single_turn_output.extra_fields["is_search"],
                    "is_terminal": single_turn_output.extra_fields["is_terminal"],
                    "search_query": single_turn_output.extra_fields["search_query"],
                    "visible_evidence": single_turn_output.extra_fields[
                        "visible_evidence"
                    ],
                    "evidence_hash": single_turn_output.extra_fields.get(
                        "evidence_hash"
                    ),
                    "format_valid": single_turn_output.extra_fields["format_valid"],
                }
            )
        output.extra_fields["turns"] = turns

        return output

    async def generate_llm_response(
        self,
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
    ):
        llm_output = None

        if generate_context["next_turn_id"] >= self.cfg.agentloop.max_turns:
            previous_output = generate_context.get("last_llm_output")
            if previous_output is not None:
                previous_output.is_end = True
                previous_output.extra_fields["is_terminal"] = True
            return False, None, None, llm_output

        # Generate response from LLM
        max_total_len = getattr(
            self, "max_total_len", len(problem_prompt_ids) + self.max_resp_len
        )
        max_resp_len = min(
            self.max_resp_len,
            max_total_len - len(turn_prompt_ids),
        )
        if max_resp_len <= 0:
            previous_output = generate_context.get("last_llm_output")
            if previous_output is not None:
                previous_output.is_end = True
                previous_output.extra_fields["is_terminal"] = True
            return False, None, None, llm_output

        generate_result = await self.generate(
            turn_prompt_ids, sampling_params={"max_new_tokens": max_resp_len}
        )
        llm_response_ids: list[int] = generate_result["output_ids"]

        if len(llm_response_ids) > max_resp_len:
            llm_response_ids = llm_response_ids[:max_resp_len]
        llm_response_text = self.tokenizer.decode(llm_response_ids)

        # split </search> manually
        if "</search>" in llm_response_text:
            llm_response_text = llm_response_text.split("</search>")[0] + "</search>"
            llm_response_ids = self.tokenizer.encode(llm_response_text)
            llm_response_ids = llm_response_ids[:max_resp_len]
            llm_response_text = self.tokenizer.decode(llm_response_ids)

        turn_id = generate_context["next_turn_id"]
        generate_context["next_turn_id"] += 1
        llm_output = AgentLoopOutput(
            prompt_ids=copy.deepcopy(turn_prompt_ids),
            response_ids=llm_response_ids,
            prompt_text=self.tokenizer.decode(turn_prompt_ids),
            response_text=llm_response_text,
            is_end=True,
            reward_score=0.0,
            extra_fields={
                "turn_id": turn_id,
                "is_search": False,
                "is_terminal": True,
                "search_query": None,
                "visible_evidence": None,
                "evidence_hash": None,
                "format_valid": "<answer>" in llm_response_text
                and "</answer>" in llm_response_text,
            },
        )
        generate_context["last_llm_output"] = llm_output
        generate_context["all_llm_response_ids"] += llm_response_ids

        if len(llm_response_ids) == max_resp_len:
            return False, None, None, llm_output

        return True, llm_response_ids, llm_response_text, llm_output

    async def generate_tool_response(
        self,
        generate_context: dict[str, Any],
        trace_prints: list[dict],
        problem_prompt_ids: list[int],
        turn_prompt_ids: list[int],
        llm_response_ids,
        llm_response_text,
    ):
        # Extract tool calls from response
        _, tool_requests = await self.toolcall_parser(llm_response_text)
        llm_output: AgentLoopOutput = generate_context["last_llm_output"]
        if tool_requests == []:
            return False, None

        search_query = str(tool_requests[-1].arguments.get("keyword", "")).strip()
        llm_output.extra_fields["is_search"] = True
        llm_output.extra_fields["search_query"] = search_query
        llm_output.extra_fields["format_valid"] = bool(search_query)
        if not search_query:
            return False, None

        # A search on the last allowed model turn cannot influence an answer.
        if generate_context["next_turn_id"] >= self.cfg.agentloop.max_turns:
            return False, None

        # Execute tools in parallel with history propagation
        tasks = []
        for tool_request in tool_requests:
            tasks.append(self.tool_call(tool_request))
        tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

        # Convert tool responses to messages and tokenize
        tool_messages = []
        for tool_response in tool_responses:
            message = {"role": "tool", "content": tool_response.text}
            tool_messages.append(message)

        # Tokenize tool responses
        tool_response_ids: list[int] = self.tokenizer.encode(
            tool_messages[0]["content"], add_special_tokens=False
        )
        next_prompt_prefix = turn_prompt_ids
        if llm_output.extra_fields["turn_id"] == 0 and generate_context.get(
            "guidance_applied", False
        ):
            # The plan may influence only the first policy search. Remove it
            # before the answer turn while retaining the policy's actual query.
            next_prompt_prefix = generate_context["unguided_problem_prompt_ids"]
        max_total_len = getattr(
            self, "max_total_len", len(problem_prompt_ids) + self.max_resp_len
        )
        available_tool_tokens = max_total_len - (
            len(next_prompt_prefix) + len(llm_response_ids)
        )
        # Reserve at least one token for the next model turn.
        max_tool_resp_len = min(
            self.max_tool_response_length, max(0, available_tool_tokens - 1)
        )
        tool_response_ids = truncate_token_ids(
            tool_response_ids,
            max_tool_resp_len,
            self.tool_response_truncate_side,
        )
        if not tool_response_ids:
            return False, None

        visible_evidence = self.tokenizer.decode(tool_response_ids)
        llm_output.is_end = False
        llm_output.extra_fields["is_terminal"] = False
        llm_output.extra_fields["visible_evidence"] = visible_evidence
        llm_output.extra_fields["evidence_hash"] = hashlib.sha256(
            visible_evidence.encode("utf-8")
        ).hexdigest()
        next_turn_prompt_ids = next_prompt_prefix + llm_response_ids + tool_response_ids
        if self.print_outputs:
            # add anything you want to print
            trace_prints.append(
                {
                    "prompt": self.tokenizer.decode(turn_prompt_ids),
                    "generate": llm_response_text,
                    "tool_resp": visible_evidence,
                }
            )
        return True, next_turn_prompt_ids

    def gen_extra_fields(self, task_results, answer):
        """Collect reward-visible text and numeric training metadata."""
        extra_fields_turn = {
            "turn_id": [],
            "is_search": [],
            "is_terminal": [],
            "search_query": [],
            "visible_evidence": [],
            "evidence_hash": [],
            "format_valid": [],
            "prompt_text": [],
            "response_text": [],
            "guidance_mode": [],
            "conditioning_group_id": [],
            "teacher_version": [],
            "teacher_plan_id": [],
            "teacher_plan_node_id": [],
        }
        extra_fields_traj = {
            "llm_reward": [],
            "response_text": [],
            "prompt_text": [],
            "turns": [],
            "sample_id": [],
            "trajectory_id": [],
            "guidance_mode": [],
            "conditioning_group_id": [],
            "teacher_version": [],
            "teacher_plan_id": [],
            "teacher_plan_node_id": [],
            "teacher_plan_valid": [],
            "teacher_plan": [],
            "teacher_plan_error": [],
            "teacher_cache_hit": [],
            "guidance_applied": [],
            "policy_version": [],
        }
        extra_fields_train = {
            "idx_to_sub_traj": [],
            "planner_turn_idx": [],
            "is_search": [],
            "is_terminal": [],
        }
        for task_result in task_results:
            extra_fields_traj["llm_reward"].append(
                task_result.extra_fields.get("llm_reward", 0.0)
            )
            extra_fields_traj["response_text"].append(
                task_result.extra_fields.get("response_text", "")
            )
            extra_fields_traj["prompt_text"].append(
                task_result.extra_fields.get("prompt_text", "")
            )
            extra_fields_traj["turns"].append(task_result.extra_fields.get("turns", []))
            for key in (
                "sample_id",
                "trajectory_id",
                "guidance_mode",
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
            ):
                extra_fields_traj[key].append(task_result.extra_fields.get(key))
            for single_turn_output in task_result.single_turn_outputs:
                metadata = single_turn_output.extra_fields
                for key in (
                    "turn_id",
                    "is_search",
                    "is_terminal",
                    "search_query",
                    "visible_evidence",
                    "evidence_hash",
                    "format_valid",
                    "guidance_mode",
                    "conditioning_group_id",
                    "teacher_version",
                    "teacher_plan_id",
                    "teacher_plan_node_id",
                ):
                    extra_fields_turn[key].append(metadata.get(key))
                extra_fields_turn["prompt_text"].append(single_turn_output.prompt_text)
                extra_fields_turn["response_text"].append(
                    single_turn_output.response_text
                )
                extra_fields_train["idx_to_sub_traj"].append(0)
                extra_fields_train["planner_turn_idx"].append(metadata["turn_id"])
                extra_fields_train["is_search"].append(metadata["is_search"])
                extra_fields_train["is_terminal"].append(metadata["is_terminal"])

        first_result = task_results[0].extra_fields if task_results else {}
        return (
            extra_fields_turn,
            extra_fields_traj,
            {
                "reference_id": answer,
                "sample_id": first_result.get("sample_id"),
                "policy_version": first_result.get("policy_version"),
                "teacher_version": first_result.get("teacher_version"),
            },
            extra_fields_train,
        )

    async def _run_shadow_rollout_group(
        self,
        input_ids: list[int],
        reference_id: str,
        question_text: str,
        sample_id: str | int,
        actual_plan: TeacherPlanResult,
        shuffled_plan: TeacherPlanResult | None,
        output_channel: Channel,
    ) -> dict:
        """Run one configured guided/unguided shadow group."""
        rollout_tasks = []
        for guidance_mode in self.guidance_modes:
            plan_result = None
            if guidance_mode in {"guided", "generic"}:
                plan_result = actual_plan
            elif guidance_mode == "shuffled":
                plan_result = shuffled_plan
            rollout_tasks.append(
                asyncio.create_task(
                    self.run_one_query(
                        copy.deepcopy(input_ids),
                        answer=reference_id,
                        question_text=question_text,
                        sample_id=sample_id,
                        guidance_mode=guidance_mode,
                        teacher_plan_result=plan_result,
                    )
                )
            )

        task_results = await asyncio.gather(*rollout_tasks)
        extra_fields = self.gen_extra_fields(task_results, reference_id)
        rollout_result = self.get_rollout_result(task_results, *extra_fields)
        agent_metrics = self.get_rollout_metrics(rollout_result)
        await output_channel.put(rollout_result, async_op=True).async_wait()
        return agent_metrics

    async def run_agentloop_rollout(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        """Run baseline groups or frozen-teacher shadow A/B groups."""
        if not self.teacher_planner_enabled:
            return await super().run_agentloop_rollout(input_channel, output_channel)
        assert self.teacher_planner is not None

        with self.worker_timer():
            rollout_request: RolloutRequest = input_channel.get()
            prompt_texts = rollout_request.prompt_texts or [None] * len(
                rollout_request.input_ids
            )
            prompt_texts = [
                prompt_text
                if prompt_text is not None
                else self.tokenizer.decode(input_ids)
                for prompt_text, input_ids in zip(
                    prompt_texts, rollout_request.input_ids, strict=True
                )
            ]
            sample_ids = rollout_request.sample_ids or list(
                range(len(rollout_request.input_ids))
            )
            if len(prompt_texts) != len(rollout_request.input_ids):
                raise ValueError("Search-R1 prompt_texts must align with input_ids")
            if len(sample_ids) != len(rollout_request.input_ids):
                raise ValueError("Search-R1 sample_ids must align with input_ids")

            actual_plans = await asyncio.gather(
                *(
                    self.teacher_planner.get_plan(question_text, self.generate)
                    for question_text in prompt_texts
                )
            )
            shuffled_plans: list[TeacherPlanResult | None] = [None] * len(actual_plans)
            if "shuffled" in self.guidance_modes:
                if len(actual_plans) < 2:
                    raise ValueError(
                        "shuffled teacher control requires at least two questions "
                        "per agent-loop request"
                    )
                shuffled_plans = shuffled_teacher_plans(
                    actual_plans, sample_ids, self.teacher_planner.seed
                )

            send_output_tasks = []
            for (
                input_ids,
                reference_id,
                question_text,
                sample_id,
                actual_plan,
                shuffled_plan,
            ) in zip(
                rollout_request.input_ids,
                rollout_request.answers,
                prompt_texts,
                sample_ids,
                actual_plans,
                shuffled_plans,
                strict=True,
            ):
                send_output_tasks.append(
                    asyncio.create_task(
                        self._run_shadow_rollout_group(
                            input_ids,
                            reference_id,
                            question_text,
                            sample_id,
                            actual_plan,
                            shuffled_plan,
                            output_channel,
                        )
                    )
                )

            agent_metrics_list = await asyncio.gather(*send_output_tasks)
            return self.post_process_metric(agent_metrics_list)
