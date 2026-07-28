# Copyright 2025 The RLinf Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
from typing import Optional
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.agents.wideseek_r2.utils.metrics import (
    get_rollout_metrics as compute_rollout_metrics,
)
from rlinf.agents.wideseek_r2.utils.prompt_utils import (
    build_message_history_and_tools,
    get_access_summary_messages,
    get_access_summary_tool_message,
    get_access_tool_message,
    get_first_turn_hint,
    get_next_turn_hint,
    get_planner_subtask_failed_message,
    get_planner_subtask_result_message,
    get_search_tool_message,
)
from rlinf.agents.wideseek_r2.utils.reward import (
    credit_assignment,
    evaluate_worker_quality,
    extract_final_answer,
    get_final_reward_score,
)
from rlinf.agents.wideseek_r2.utils.sglang_client import SGLangClient
from rlinf.agents.wideseek_r2.utils.utils import (
    _build_tool_call_info,
    _set_max_turns,
    populate_turn_extra_fields,
)
from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.data.tool_call.tool_io_struct import (
    ToolRequest,
    ToolResponse,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import (
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)


class WideSeekR2AgentLoopWorker(MultiAgentLoopWorker):
    """Multi-turn WideSeek-R2 agent worker for MAS and single-agent workflows."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.extra_keys_turn = [
            "subtask_count",
            "search_count",
            "access_count",
            "tool_call_info",
            "prompt_text",
            "response_text",
            "role",
        ]
        self.extra_keys_traj = [
            "origin_question",
            "final_answer",
            "final_answer_text",
            "num_valid_planner_turns",
            "num_valid_worker_turns",
            "total_turn_list",
            "final_answer_format",
            "llm_reward",
            "gisa_metrics",
        ]

        self.max_prompt_len = int(self.cfg.data.max_prompt_length)
        self.max_total_len = int(self.cfg.runner.seq_length)

        self.use_access_summary = self.cfg.tools.get("use_access_summary", False)
        self.use_llm_judge = self.cfg.agentloop.get("use_llm_judge", True)

        self.placement = placement
        self.use_fixed_rollout = cfg.rollout.get("use_fixed_worker", False)
        self.fixed_role = self.cfg.agentloop.get("fixed_role", None)
        if self.use_fixed_rollout:
            assert self.fixed_role

        self.workflow = self.cfg.agentloop.get("workflow", "mas")

        if self.use_llm_judge:
            llm_ip = self.cfg.agentloop.get("llm_ip", "")
            llm_port = self.cfg.agentloop.get("llm_port", "")
            llm_type = self.cfg.agentloop.get("llm_type", "")
            self.sgl_client = SGLangClient(llm_ip, llm_port, llm_type)
            self.use_local_judge = self.cfg.agentloop.get("use_local_judge", False)
            if self.use_local_judge:
                self.llm_generator = self.local_judge_llm_generator
            else:
                self.llm_generator = self.sgl_client.call_sglang_api

        else:
            self.sgl_client = None
            self.llm_generator = None

        assert self.return_logprobs if not self.is_eval else True

        assert self.toolcall_parser is not None, (
            "toolcall_parser must be set in wideseek_r2"
        )

    async def local_judge_llm_generator(self, messages: list) -> str:
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        # invocate generate method
        generate_result = await self.generate(
            prompt_ids,
            rollout_name="rollout_judge",
        )

        # decode generate_result["output_ids"] to judge_response_text
        judge_response_text = self.tokenizer.decode(generate_result["output_ids"])
        return judge_response_text

    async def extract_tool_calls(
        self, response_text: str, role: str
    ) -> tuple[list[ToolRequest], Optional[dict]]:
        """Parse tool calls via the registered parser and build turn metrics.

        Args:
            response_text: Decoded model response that may contain tool-call JSON.
            role: Current role (`planner`, `worker`, or `single`).

        Returns:
            A tuple of `(tool_requests, tool_call_info)` where `tool_call_info`
            summarizes subtask/search/access counts for metrics.
        """
        max_workers_per_planner = self.cfg.agentloop.get("max_workers_per_planner", -1)
        max_toolcall_per_worker = self.cfg.agentloop.get("max_toolcall_per_worker", 5)
        assert self.toolcall_parser is not None
        _, tool_requests = await self.toolcall_parser(
            response_text,
            role=role,
            max_workers_per_planner=max_workers_per_planner,
            max_toolcall_per_worker=max_toolcall_per_worker,
        )
        tool_call_info = _build_tool_call_info(role=role, tool_requests=tool_requests)
        return tool_requests, tool_call_info

    async def access_sumamry(self, info_to_extract, page_content):
        """Summarize access content to keep context compact for follow-up turns.

        Args:
            info_to_extract: Focus information requested by the worker.
            page_content: Raw page text returned by the access tool.

        Returns:
            A short summary string for tool feedback.
        """
        if not self.use_llm_judge:
            return page_content

        if page_content == "No More Information is Found for this URL.":
            return "No useful Information is Found under this URL."

        messages = get_access_summary_messages(info_to_extract, page_content)
        result_text = await self.llm_generator(messages)
        return result_text

    def _build_role_context(
        self,
        *,
        origin_question: str,
        role: str,
        add_few_shot: bool,
        max_workers_per_planner: int,
        max_toolcall_per_worker: int,
        main_task: str | None,
        answer_type: str | None,
        max_turns: int,
    ) -> list[int]:
        """Build the initial prompt for one role conversation.

        This hook deliberately contains the existing prompt construction logic
        unchanged.  Workflow-specific loop implementations can override it to
        add scoped context while keeping the base MAS/SA prompt behavior
        stable.

        Args:
            origin_question: Role-specific question.
            role: Current role (`planner`, `worker`, or `single`).
            add_few_shot: Whether to include few-shot examples.
            max_workers_per_planner: Per-turn planner fan-out limit.
            max_toolcall_per_worker: Per-turn worker tool-call limit.
            main_task: Original task text for a worker, if any.
            answer_type: Dataset-provided answer structure.
            max_turns: Maximum number of turns for the role.

        Returns:
            Tokenized and length-limited initial prompt ids.
        """
        message_history, tools = build_message_history_and_tools(
            origin_question=origin_question,
            role=role,
            add_few_shot=add_few_shot,
            max_workers_per_planner=max_workers_per_planner,
            max_toolcall_per_worker=max_toolcall_per_worker,
            main_task=main_task,
            answer_type=answer_type,
        )

        turn_hint = get_first_turn_hint(max_turns=max_turns)
        assert message_history[-1]["role"] == "user"
        message_history[-1]["content"] += turn_hint

        prompt_ids = self.tokenizer.apply_chat_template(
            message_history, tokenize=True, add_generation_prompt=True, tools=tools
        )
        return prompt_ids[: self.max_total_len]

    async def _before_role_turn(
        self,
        *,
        prompt_ids: list[int],
        role: str,
        turn_idx: int,
        conv_id: str,
    ) -> list[int]:
        """Update a role prompt immediately before generation.

        The default implementation is a no-op.  Subclasses can use this hook
        to inject turn-boundary events without changing the base workflow.

        Args:
            prompt_ids: Current conversation prompt ids.
            role: Current role (`planner`, `worker`, or `single`).
            turn_idx: Zero-based turn index.
            conv_id: Stable conversation id used for rollout affinity.

        Returns:
            Prompt ids to use for the current generation.
        """
        del role, turn_idx, conv_id
        return prompt_ids

    async def _dispatch_planner_requests(
        self,
        tool_requests: list[ToolRequest],
        *,
        main_task: str,
        sub_traj_id: int,
        sub_traj_num: int,
    ) -> list[tuple[list[AgentLoopOutput], Optional[str], list]]:
        """Run planner-created worker requests concurrently.

        This is the default planner fan-out path.  It preserves the existing
        `asyncio.gather` semantics so an unhandled worker error still aborts
        the query, while allowing a graph workflow to provide a scheduler-
        backed implementation later.

        Args:
            tool_requests: Parsed planner subtask requests.
            main_task: Original user question.
            sub_traj_id: Planner sub-trajectory id (normally zero).
            sub_traj_num: Number of worker trajectories already allocated in
                this planner conversation.

        Returns:
            Results in the same order as `tool_requests`.
        """
        assert sub_traj_id == 0
        tasks = [
            self.worker_call(
                tool_request,
                main_task,
                sub_traj_id + index + 1 + sub_traj_num,
            )
            for index, tool_request in enumerate(tool_requests)
        ]
        return await asyncio.gather(*tasks)

    def _format_planner_feedback(
        self,
        tool_requests: list[ToolRequest],
        worker_results: list[tuple[list[AgentLoopOutput], Optional[str], list]],
        *,
        turn_idx: int,
        max_turns: int,
    ) -> tuple[list[AgentLoopOutput], list, list[dict]]:
        """Format worker results for the next planner turn.

        Args:
            tool_requests: Planner requests corresponding to `worker_results`.
            worker_results: Worker outputs in request order.
            turn_idx: Current planner turn index.
            max_turns: Planner turn budget.

        Returns:
            Worker output buffer, worker turn statistics, and tool messages for
            the planner conversation.
        """
        worker_buffer = []
        worker_turn_list = []
        tool_messages_text = []
        for idx, (
            worker_outputs_buffer,
            worker_summary,
            total_turn_list_worker,
        ) in enumerate(worker_results):
            worker_buffer.extend(worker_outputs_buffer)
            worker_turn_list.extend(total_turn_list_worker)
            # Format tool response with both request and result.
            subtask_text = tool_requests[idx].arguments["subtask"]
            # The worker `<answer>` extraction result decides success:
            # a present summary -> result message, None -> failed message.
            if worker_summary is not None:
                tool_messages_text.append(
                    get_planner_subtask_result_message(
                        subtask_idx=idx + 1,
                        subtask_text=subtask_text,
                        worker_summary=worker_summary,
                    )
                )
            else:
                tool_messages_text.append(
                    get_planner_subtask_failed_message(
                        subtask_idx=idx + 1,
                        subtask_text=subtask_text,
                    )
                )

        turn_hint = get_next_turn_hint(
            next_turn_idx=turn_idx + 2,
            max_turns=max_turns,
        )
        tool_messages = [
            {
                "role": "tool",
                "content": "\n\n".join(tool_messages_text) + turn_hint,
            }
        ]
        return worker_buffer, worker_turn_list, tool_messages

    async def _dispatch_worker_requests(
        self, tool_requests: list[ToolRequest]
    ) -> list[ToolResponse]:
        """Execute worker or single-agent external tools concurrently."""
        return await asyncio.gather(
            *(self.tool_call(tool_request) for tool_request in tool_requests)
        )

    async def _format_worker_feedback(
        self,
        tool_requests: list[ToolRequest],
        tool_responses: list[ToolResponse],
        *,
        turn_idx: int,
        max_turns: int,
    ) -> list[dict]:
        """Format external tool responses for the next role turn.

        Access summaries intentionally remain in this hook so a future graph
        workflow can replace the feedback channel without changing external
        tool execution.
        """
        tool_messages_text = []
        access_summary_jobs = []
        for idx, (tool_request, tool_response) in enumerate(
            zip(tool_requests, tool_responses)
        ):
            # Include the original request and the result.
            if tool_request.name == "search":
                query = tool_request.arguments["query"]
                tool_messages_text.append(
                    get_search_tool_message(
                        query=query,
                        search_result=tool_response.text,
                    )
                )
            elif tool_request.name == "access":
                url = tool_request.arguments["url"]
                info_to_extract = tool_request.arguments["info_to_extract"]
                page_content = tool_response.text
                if self.use_access_summary:
                    tool_messages_text.append(None)
                    coro = self.access_sumamry(info_to_extract, page_content)
                    access_summary_jobs.append((idx, url, info_to_extract, coro))
                else:
                    tool_messages_text.append(
                        get_access_tool_message(
                            url=url,
                            page_content=page_content,
                        )
                    )
            else:
                raise ValueError(f"Unknown tool request name: {tool_request.name}")

        if self.use_access_summary and access_summary_jobs:
            coros = [job[-1] for job in access_summary_jobs]
            summaries = await asyncio.gather(*coros)
            for job, summary in zip(access_summary_jobs, summaries):
                idx, url, info_to_extract, _ = job
                tool_messages_text[idx] = get_access_summary_tool_message(
                    url=url,
                    info_to_extract=info_to_extract,
                    summary=summary,
                )

        turn_hint = get_next_turn_hint(
            next_turn_idx=turn_idx + 2,
            max_turns=max_turns,
        )
        return [
            {
                "role": "tool",
                "content": "\n\n".join(tool_messages_text) + turn_hint,
            }
        ]

    async def _finalize_trajectory(
        self,
        *,
        role: str,
        response_text: str,
        turn_idx: int,
        total_turn_list: list,
        conv_id: str,
    ) -> tuple[Optional[str], list]:
        """Extract the role result and release conversation affinity.

        The default implementation is the existing WideSeek-R2 finalization
        path.  It is a hook because graph workflows will eventually finalize
        through audit and deterministic rendering.
        """
        if role == "worker":
            # Workers must return their final answer inside <answer>...</answer>;
            # a missing tag yields None, which routes to the failed planner
            # message.
            answer_text = extract_final_answer(response_text, mode="tag")
        else:
            answer_text = response_text.split("<|im_end|>")[0]

        total_turn_list.append(turn_idx + 1)
        # Release this conversation's load from the affinity router (no-op
        # unless the load-aware policy is active). Per-query exceptions abort
        # the whole rollout, so the normal return path releases here.
        self.release_affinity(conv_id)
        return answer_text, total_turn_list

    async def _after_role_loop(
        self,
        *,
        prompt_ids: list[int],
        response_text: str,
        role: str,
        sub_traj_id: int,
        turn_idx: int,
        conv_id: str,
        output_buffer: list[AgentLoopOutput],
    ) -> tuple[str, int]:
        """Optionally add a workflow-specific terminal generation."""

        del prompt_ids, role, sub_traj_id, conv_id, output_buffer
        return response_text, turn_idx

    async def worker_call(
        self,
        worker_request: ToolRequest,
        main_task: str,
        sub_traj_id: int,
    ) -> tuple[list[AgentLoopOutput], Optional[str], list]:
        """Execute one planner-created subtask through the worker role loop.

        Args:
            worker_request: Planner output converted to a `subtask` tool request.
            main_task: Original user question for worker grounding.
            sub_traj_id: Sub-trajectory index used for training regrouping.

        Returns:
            Worker turn outputs, extracted summary, and worker turn statistics.
        """
        assert worker_request.name == "subtask", (
            f"Expected 'subtask' tool, got {worker_request.name}"
        )
        assert "subtask" in worker_request.arguments, (
            f"Missing 'subtask' in arguments: {worker_request.arguments}"
        )
        assert sub_traj_id > 0
        subtask = worker_request.arguments["subtask"]

        (
            worker_outputs_buffer,
            answer_text,
            total_turn_list,
        ) = await self.run_one_query_role(
            question=subtask,
            role="worker",
            sub_traj_id=sub_traj_id,
            main_task=main_task,
        )
        worker_format_valid = bool(answer_text and answer_text.strip())
        quality_score = 0.0
        quality_valid = False
        if (
            self.cfg.agentloop.get("agent_level_credit_assignment", False)
            and self.cfg.agentloop.get("worker_quality_judge", True)
            and not self.is_eval
        ):
            max_chars = self.cfg.agentloop.get(
                "worker_quality_max_context_chars", 12000
            )
            evidence_context = (
                worker_outputs_buffer[-1].prompt_text[-max_chars:]
                if worker_outputs_buffer
                else ""
            )
            quality_score, quality_valid = await evaluate_worker_quality(
                main_question=main_task,
                subtask=subtask,
                worker_summary=answer_text,
                evidence_context=evidence_context,
                judge_llm_generator=self.llm_generator,
            )
        for turn in worker_outputs_buffer:
            turn.extra_fields["worker_quality_score"] = quality_score
            turn.extra_fields["worker_quality_valid"] = quality_valid
            turn.extra_fields["worker_format_valid"] = worker_format_valid
        return worker_outputs_buffer, answer_text, total_turn_list

    async def run_one_query_role(
        self,
        question: str,
        role: str,
        sub_traj_id: int,
        main_task: str | None = None,
        answer_type: str | None = None,
    ) -> tuple[list[AgentLoopOutput], Optional[str], list]:
        """Run one query under a specific role until stop or turn budget.

        Args:
            question: Role-specific input question (main query or subtask).
            role: One of `planner`, `worker`, or `single`.
            sub_traj_id: Sub-trajectory id for downstream regrouping.
            main_task: Original task text required when `role == "worker"`.
            answer_type: Dataset-provided answer structure for no-shot main roles.

        Returns:
            Tuple of `(output_buffer, answer_text, total_turn_list)`. For workers,
            `answer_text` is the extracted `<answer>` content, or None when the
            worker did not produce a valid answer block.
        """

        origin_question = question
        output_buffer = []
        total_turn_list = []

        # Stable per-conversation id: every turn of this role loop reuses it so
        # they are consistently hashed to the same SGLang worker (KV cache reuse).
        conv_id = uuid4().hex

        add_few_shot = self.cfg.agentloop.get("add_few_shot", True)
        max_workers_per_planner = self.cfg.agentloop.get("max_workers_per_planner", -1)
        max_toolcall_per_worker = self.cfg.agentloop.get("max_toolcall_per_worker", 5)
        max_turns = _set_max_turns(self.cfg.agentloop, role)

        prompt_ids = self._build_role_context(
            origin_question=origin_question,
            role=role,
            add_few_shot=add_few_shot,
            max_workers_per_planner=max_workers_per_planner,
            max_toolcall_per_worker=max_toolcall_per_worker,
            main_task=main_task,
            answer_type=answer_type,
            max_turns=max_turns,
        )

        response_text = ""
        sub_traj_num = 0

        turn_idx = -1
        for turn_idx in range(max_turns):
            prompt_ids = await self._before_role_turn(
                prompt_ids=prompt_ids,
                role=role,
                turn_idx=turn_idx,
                conv_id=conv_id,
            )
            max_resp_len = self.max_total_len - len(prompt_ids)
            if max_resp_len <= 0:
                break

            if role == self.fixed_role and self.use_fixed_rollout:
                generate_result = await self.generate(
                    prompt_ids,
                    sampling_params={"max_new_tokens": max_resp_len},
                    rollout_name="subworker",
                    session_id=conv_id,
                )
                generate_result["logprobs"] = [0.0] * len(generate_result["output_ids"])
            else:
                generate_result = await self.generate(
                    prompt_ids,
                    sampling_params={"max_new_tokens": max_resp_len},
                    session_id=conv_id,
                )

            response_ids = generate_result["output_ids"]
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]

            response_text = self.tokenizer.decode(response_ids)

            tool_requests, tool_call_info = await self.extract_tool_calls(
                response_text, role=role
            )

            output_buffer.append(
                AgentLoopOutput(
                    prompt_ids=copy.deepcopy(prompt_ids),
                    response_ids=copy.deepcopy(response_ids),
                    prompt_text=copy.deepcopy(self.tokenizer.decode(prompt_ids)),
                    response_text=response_text,
                    is_end=generate_result["finish_reason"] == "length",
                    response_logprobs=generate_result["logprobs"]
                    if self.return_logprobs
                    else None,
                    extra_fields={
                        "role": role,
                        "idx_to_sub_traj": sub_traj_id,
                        "worker_quality_score": 0.0,
                        "worker_quality_valid": False,
                        "worker_format_valid": False,
                    },
                    tool_call_info=tool_call_info
                    if tool_call_info
                    else None,  # if passed, must have tool call
                )
            )

            prompt_ids += response_ids

            if len(response_ids) == max_resp_len:
                break

            # Extract tool calls
            if tool_requests == []:
                break

            # Handle tool calls based on role.
            worker_buffer = []
            worker_turn_list = []
            if role == "planner":
                assert sub_traj_id == 0
                # Planner fans out multiple sub-agents in parallel.
                sub_traj_num_before_dispatch = sub_traj_num
                sub_traj_num += len(tool_requests)
                worker_results = await self._dispatch_planner_requests(
                    tool_requests,
                    main_task=origin_question,
                    sub_traj_id=sub_traj_id,
                    sub_traj_num=sub_traj_num_before_dispatch,
                )
                (
                    worker_buffer,
                    worker_turn_list,
                    tool_messages,
                ) = self._format_planner_feedback(
                    tool_requests,
                    worker_results,
                    turn_idx=turn_idx,
                    max_turns=max_turns,
                )

            else:
                # Worker/single executes search/access tools in parallel.
                tool_responses = await self._dispatch_worker_requests(tool_requests)
                tool_messages = await self._format_worker_feedback(
                    tool_requests,
                    tool_responses,
                    turn_idx=turn_idx,
                    max_turns=max_turns,
                )

            # Tokenize tool responses
            tool_response_ids = self.get_tool_response_ids(tool_messages)
            max_tool_resp_len = self.max_total_len - len(prompt_ids)
            if len(tool_response_ids) >= max_tool_resp_len:
                break

            prompt_ids += tool_response_ids
            output_buffer.extend(worker_buffer)
            total_turn_list.extend(worker_turn_list)

        response_text, turn_idx = await self._after_role_loop(
            prompt_ids=prompt_ids,
            response_text=response_text,
            role=role,
            sub_traj_id=sub_traj_id,
            turn_idx=turn_idx,
            conv_id=conv_id,
            output_buffer=output_buffer,
        )
        answer_text, total_turn_list = await self._finalize_trajectory(
            role=role,
            response_text=response_text,
            turn_idx=turn_idx,
            total_turn_list=total_turn_list,
            conv_id=conv_id,
        )
        return output_buffer, answer_text, total_turn_list

    async def run_one_query(self, prompt_ids: list[int], *, answer) -> AgentLoopOutput:
        """Run one sample end-to-end and attach reward/training metadata.

        Args:
            prompt_ids: Tokenized query prompt from the dataset.
            answer: Label payload used for format extraction and reward scoring.

        Returns:
            A multi-turn output object containing all turns and trajectory metadata.
        """
        sub_traj_id = 0
        origin_question = self.tokenizer.decode(prompt_ids)
        if self.workflow == "sa":
            role = "single"
        else:
            role = "planner"

        answer_type = answer.get("answer_type")

        (
            output_buffer,
            answer_text,
            total_turn_list,
        ) = await self.run_one_query_role(
            question=origin_question,
            role=role,
            sub_traj_id=sub_traj_id,
            answer_type=answer_type,
        )

        final_answer_extract = extract_final_answer(
            answer_text, mode="markdown", strict=False
        )

        # credit assignment
        norm_column = self.cfg.data.get("norm_column", False)
        llm_reward, format, gisa_metrics = await get_final_reward_score(
            origin_question,
            final_answer_extract,
            answer,
            norm_column,
            self.llm_generator,
        )

        answer_format = final_answer_extract is not None and format is True
        reward_score = credit_assignment(
            agentloop_config=self.cfg.agentloop,
            llm_reward=llm_reward,
            answer_format=answer_format,
        )
        final_answer_format = 1 if answer_format else 0

        # In wideseek_r2 every generated turn is trainable: assign the trajectory
        # reward to all turns and mark them for training. The shared not_training
        # filter in agent_loop becomes a no-op because it is always False here.
        for single_turn_output in output_buffer:
            single_turn_output.reward_score = reward_score
            single_turn_output.extra_fields["not_training"] = False

        num_valid_planner_turns, num_valid_worker_turns = populate_turn_extra_fields(
            output_buffer
        )

        output = MultiAgentLoopOutput(
            single_turn_outputs=output_buffer,
            trace_prints=[],  # Can add message_history tracking if needed
            extra_fields={
                "final_answer": final_answer_extract,
                "final_answer_text": answer_text,
                "reward": reward_score,
                "origin_question": origin_question,
                "llm_reward": llm_reward,
                "gisa_metrics": gisa_metrics,
                "total_turn_list": total_turn_list
                if self.workflow in {"mas", "mas_graph"}
                else None,
                "instance_id": answer["instance_id"],
                "num_valid_planner_turns": num_valid_planner_turns,
                "num_valid_worker_turns": num_valid_worker_turns,
                "final_answer_format": final_answer_format,
            },
        )
        return output

    def gen_extra_fields(
        self,
        task_results: list[MultiAgentLoopOutput],
        answer: str,
    ) -> Optional[dict]:
        """Build extra fields for turn/traj/group scopes and training regrouping.

        Args:
            task_results: Grouped rollout samples for the same input question.
            answer: Ground-truth answer payload for this group.

        Returns:
            Extra field dicts for turn-level, trajectory-level, group-level,
            and training-only fields.
        """
        extra_fields_turn, extra_fields_traj, *_ = super().gen_extra_fields(
            task_results, answer
        )

        roles = []
        for task_result in task_results:
            for single_turn_output in task_result.single_turn_outputs:
                if self.extra_keys_turn is not None:
                    for k in self.extra_keys_turn:
                        v = single_turn_output.extra_fields.get(k, None)
                        if (
                            k == "role"
                            and not single_turn_output.extra_fields["not_training"]
                        ):
                            roles.append(v)
        extra_fields_turn = {**extra_fields_turn, "roles": roles}

        extra_fields_group = {
            "answer": answer,
            "num_valid_planner_turns": sum(
                extra_fields_traj["num_valid_planner_turns"]
            ),
            "num_valid_worker_turns": sum(extra_fields_traj["num_valid_worker_turns"]),
        }

        idx_to_sub_traj = []
        role_ids = []
        worker_quality_score = []
        worker_quality_valid = []
        worker_format_valid = []
        llm_rewards = []
        role_to_id = {"planner": 0, "worker": 1, "single": 2}
        for task_result in task_results:
            sub_traj_map = {}
            trajectory_llm_reward = task_result.extra_fields["llm_reward"]
            for single_turn_output in task_result.single_turn_outputs:
                if single_turn_output.extra_fields["not_training"]:
                    continue
                role_idx = single_turn_output.extra_fields["idx_to_sub_traj"]
                if role_idx not in sub_traj_map:
                    sub_traj_map[role_idx] = len(sub_traj_map)
                idx_to_sub_traj.append(sub_traj_map[role_idx])
                role_ids.append(role_to_id[single_turn_output.extra_fields["role"]])
                worker_quality_score.append(
                    single_turn_output.extra_fields["worker_quality_score"]
                )
                worker_quality_valid.append(
                    single_turn_output.extra_fields["worker_quality_valid"]
                )
                worker_format_valid.append(
                    single_turn_output.extra_fields["worker_format_valid"]
                )
                llm_rewards.append(trajectory_llm_reward)
        extra_fields_train = {
            "idx_to_sub_traj": idx_to_sub_traj,
            "role_id": role_ids,
            "worker_quality_score": worker_quality_score,
            "worker_quality_valid": worker_quality_valid,
            "worker_format_valid": worker_format_valid,
            "llm_reward": llm_rewards,
        }

        return (
            extra_fields_turn,
            extra_fields_traj,
            extra_fields_group,
            extra_fields_train,
        )

    def get_rollout_metrics(
        self,
        rollout_result: DynamicRolloutResult,
    ) -> dict:
        """Compute wideseek rollout metrics from packed dynamic rollout outputs.

        Args:
            rollout_result: Dynamic rollout structure produced by this worker.

        Returns:
            Aggregated metric dictionary for logging.
        """
        return compute_rollout_metrics(rollout_result, self.is_eval)
