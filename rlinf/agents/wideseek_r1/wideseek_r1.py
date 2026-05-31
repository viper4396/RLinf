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
import re
from typing import Optional

from omegaconf import DictConfig

from rlinf.agents.wideseek_r1.utils.metrics import _compute_rollout_metrics
from rlinf.agents.wideseek_r1.utils.prompt_utils import (
    get_access_summary_messages,
    get_access_summary_tool_message,
    get_access_tool_message,
    get_first_turn_hint,
    get_next_turn_hint,
    get_planner_subtask_failed_message,
    get_planner_subtask_result_message,
    get_prompt_planner,
    get_prompt_single_agent,
    get_prompt_worker,
    get_search_tool_message,
)
from rlinf.agents.wideseek_r1.utils.reward import (
    _compute_traj_bonuses,
    compute_hind_weights,
    credit_assignment,
    evaluate_turn_rewards,
    extract_final_answer,
    get_final_reward_score,
)
from rlinf.agents.wideseek_r1.utils.sglang_client import SGLangClient
from rlinf.agents.wideseek_r1.utils.tool_description import (
    tools_description_en,
    tools_description_zh,
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


class WideSeekR1AgentLoopWorker(MultiAgentLoopWorker):
    """Multi-turn WideSeek-R1 agent worker for MAS and single-agent workflows."""

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
        self.is_hybrid = self.cfg.data.get("is_hybrid", False)

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
            "toolcall_parser must be set in wideseek_r1"
        )

    @staticmethod
    def _build_tool_call_info(
        role: str, tool_requests: list[ToolRequest]
    ) -> Optional[dict]:
        if not tool_requests:
            return None

        subtask_count = 0
        search_count = 0
        access_count = 0
        for request in tool_requests:
            if request.name == "subtask":
                subtask_count += 1
            elif request.name == "search":
                search_count += 1
            elif request.name == "access":
                access_count += 1
        return {
            "subtask": subtask_count,
            "search": search_count,
            "access": access_count,
            "role": role,
        }

    @staticmethod
    def _check_final_answer(response_text: str) -> tuple[bool, bool]:
        """Check if response contains ``<answer>...</answer>`` tags.

        Returns:
            ``(has_answer, format_valid)`` where ``has_answer`` is True when
            the tags are present, and ``format_valid`` is True when the
            content between the tags is non-empty.
        """
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, response_text, re.DOTALL)
        if not matches:
            return False, False
        content = matches[-1].strip()
        return True, bool(content)

    async def compute_token_logprobs(
        self,
        prompt_text: str,
        target_token_ids: list[int],
        temperature: float = 1.0,
    ) -> list[float]:
        """Compute per-token logprobs of *target_token_ids* given *prompt_text*.

        Concatenates prompt + target tokens, asks the rollout engine to
        generate 1 extra token, and returns the logprobs for the target
        portion of the sequence.

        Args:
            prompt_text: The conditioning text (state + hindsight question).
            target_token_ids: Token IDs of the action to score.
            temperature: Sampling temperature for logprob computation.

        Returns:
            Per-token log probabilities (same length as *target_token_ids*).
        """
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = prompt_ids + list(target_token_ids)
        full_ids = full_ids[: self.max_total_len - 1]

        generate_result = await self.generate(
            full_ids,
            sampling_params={
                "max_new_tokens": 1,
                "temperature": temperature,
            },
        )

        raw_logprobs = generate_result.get("logprobs", [])
        if raw_logprobs is None:
            raw_logprobs = []

        # logprobs covers the last len(target)+1 tokens (target + generated token).
        n_target = len(target_token_ids) + 1
        target_logprobs = (
            raw_logprobs[-n_target:-1] if len(raw_logprobs) >= n_target
            else raw_logprobs
        )
        # Pad / truncate to exact length.
        if len(target_logprobs) < len(target_token_ids):
            target_logprobs = [0.0] * (len(target_token_ids) - len(target_logprobs)) + target_logprobs
        return target_logprobs[:len(target_token_ids)]

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
        max_workers_per_planner = self.cfg.agentloop.get("max_workers_per_planner", 10)
        max_toolcall_per_worker = self.cfg.agentloop.get("max_toolcall_per_worker", 5)
        assert self.toolcall_parser is not None
        _, tool_requests = await self.toolcall_parser(
            response_text,
            role=role,
            max_workers_per_planner=max_workers_per_planner,
            max_toolcall_per_worker=max_toolcall_per_worker,
        )
        tool_call_info = self._build_tool_call_info(
            role=role, tool_requests=tool_requests
        )
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

    async def worker_call(
        self,
        worker_request: ToolRequest,
        main_task: str,
        is_markdown: bool,
        language: str,
        sub_traj_id: int,
    ) -> tuple[list[AgentLoopOutput], str]:
        """Execute one planner-created subtask through the worker role loop.

        Args:
            worker_request: Planner output converted to a `subtask` tool request.
            main_task: Original user question for worker grounding.
            is_markdown: Whether this sample expects markdown-table final answers.
            language: Prompt language (`en` or `zh`).
            sub_traj_id: Sub-trajectory index used for training regrouping.

        Returns:
            Worker turn outputs, worker summary text, turn statistics, and failure flag.
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
            _,
            _,
            _,
            task_failed,
            _,
        ) = await self.run_one_query_role(
            question=subtask,
            role="worker",
            sub_traj_id=sub_traj_id,
            main_task=main_task,
            is_markdown=is_markdown,
            language=language,
        )
        return worker_outputs_buffer, answer_text, total_turn_list, task_failed

    def _set_max_turns(self, role: str) -> int:
        if role == "planner":
            return self.cfg.agentloop.get("max_planner_turns", 10)
        if role == "single":
            return self.cfg.agentloop.get("max_sa_turns", 50)
        if role == "worker":
            return self.cfg.agentloop.get("max_worker_turns", 20)
        raise ValueError(f"illegal role {role}")

    def _set_max_allow_turns(self, role: str) -> int:
        """Hard turn limit (>= soft limit).  Turns beyond the soft limit are
        marked ``max_turn_limit_failed`` but the agent is allowed to continue."""
        if role == "planner":
            return self.cfg.agentloop.get(
                "max_allow_planner_turns",
                self.cfg.agentloop.get("max_planner_turns", 10),
            )
        if role == "single":
            return self.cfg.agentloop.get(
                "max_allow_sa_turns",
                self.cfg.agentloop.get("max_sa_turns", 50),
            )
        if role == "worker":
            return self.cfg.agentloop.get(
                "max_allow_worker_turns",
                self.cfg.agentloop.get("max_worker_turns", 20),
            )
        raise ValueError(f"illegal role {role}")

    def _build_message_history_and_tools(
        self,
        origin_question: str,
        role: str,
        is_markdown: bool,
        language: str,
        main_task: str | None,
    ) -> tuple[list[dict], list[dict]]:
        """Build role-specific prompt history and exposed tool descriptions.

        Args:
            origin_question: Query text for this role loop.
            role: Current role (`planner`, `worker`, or `single`).
            is_markdown: Whether markdown answer format is required.
            language: Prompt language identifier (`en` or `zh`).
            main_task: Parent task text required for worker prompts.

        Returns:
            A tuple of `(message_history, tools)` for chat-template rendering.
        """
        tools_description = (
            tools_description_zh if language == "zh" else tools_description_en
        )
        if role == "planner":
            message_history = get_prompt_planner(
                origin_question, is_markdown=is_markdown, language=language
            )
            tools = [tools_description["create_sub_agents"]]
        elif role == "worker":
            assert main_task is not None, "Worker must have main_task provided"
            message_history = get_prompt_worker(
                main_task, origin_question, language=language
            )
            tools = [tools_description["search"], tools_description["access"]]
        elif role == "single":
            message_history = get_prompt_single_agent(
                origin_question, is_markdown=is_markdown, language=language
            )
            tools = [
                tools_description["search_single_agent"],
                tools_description["access_single_agent"],
            ]
        else:
            raise ValueError(f"Invalid role: {role}")
        return message_history, tools

    def _mark_role_failed_turns(
        self,
        *,
        output_buffer: list[AgentLoopOutput],
        role: str,
        turn_idx: int,
        succ_end: bool,
        context_failed: bool,
        tool_response_failed: bool,
        answer_format_failed: bool = False,
    ) -> bool:
        """Apply failure flags to turns for one role and return task failure.

        ``max_turn_limit_failed`` is already set per-turn inside the main loop
        (True when ``turn_idx >= max_turns``).  This method only marks
        context / tool-response / answer-format failures and computes the
        overall task-failure flag.

        Args:
            output_buffer: Collected per-turn outputs for this role execution.
            role: Current role whose turns should be marked.
            turn_idx: Last executed loop index (zero-based).
            succ_end: Whether the role loop ended via a valid answer tag.
            context_failed: Whether prompt/response length hit context limit.
            tool_response_failed: Whether tool feedback exceeded available space.
            answer_format_failed: Whether the final turn lacks valid answer tags.

        Returns:
            Boolean task failure indicator for this role execution.
        """
        if context_failed or tool_response_failed:
            for turn in output_buffer:
                if turn.extra_fields["role"] == role:
                    turn.extra_fields["context_failed"] = True

        if answer_format_failed:
            for turn in output_buffer:
                if turn.extra_fields["role"] == role:
                    turn.extra_fields["answer_format_failed"] = True

        if (
            context_failed
            and len(output_buffer) >= 1
            and len(output_buffer[-1].response_ids) >= 8000
        ):
            output_buffer[-1].extra_fields["turn_repeat_failed"] = True

        task_failed = not succ_end
        assert task_failed != succ_end
        return task_failed

    async def run_one_query_role(
        self,
        question: str,
        role: str,
        sub_traj_id: int,
        main_task: str | None = None,
        is_markdown: bool = False,
        language: str = "en",
    ) -> tuple[list[AgentLoopOutput], str, list[int], list[int], list[float], list[int], bool, bool]:
        """Run one query under a specific role until stop, failure, or turn budget.

        Args:
            question: Role-specific input question (main query or subtask).
            role: One of `planner`, `worker`, or `single`.
            sub_traj_id: Sub-trajectory id for downstream regrouping.
            main_task: Original task text required when `role == "worker"`.
            is_markdown: Whether markdown answer format is required.
            language: Prompt language.

        Returns:
            Tuple of `(output_buffer, answer_text, total_turn_list, num_turn_subagents, num_effective_subagents, access_search_ratio, task_failed, succ_end)`.
        """

        origin_question = question
        output_buffer = []
        total_turn_list = []
        num_turn_subagents = []
        num_effective_subagents = []
        access_search_ratio = []

        # Planner turn counter: increments each time the planner/single-agent
        # executes a new turn.  Workers inherit their parentʼs counter.
        planner_turn_counter = -1

        message_history, tools = self._build_message_history_and_tools(
            origin_question=origin_question,
            role=role,
            is_markdown=is_markdown,
            language=language,
            main_task=main_task,
        )
        max_turns = self._set_max_turns(role=role)
        max_allow_turns = self._set_max_allow_turns(role=role)

        turn_hint = get_first_turn_hint(max_turns=max_allow_turns, language=language)
        assert message_history[-1]["role"] == "user"
        message_history[-1]["content"] = message_history[-1]["content"] + turn_hint

        prompt_ids = self.tokenizer.apply_chat_template(
            message_history, tokenize=True, add_generation_prompt=True, tools=tools
        )
        prompt_ids = prompt_ids[: self.max_total_len]

        # Initialize tracking variables
        context_failed = False
        tool_response_failed = False
        answer_format_failed = False
        has_answer_tag = False

        succ_end = False
        sub_traj_num = 0

        turn_idx = -1
        for turn_idx in range(max_allow_turns):
            planner_turn_counter += 1
            max_resp_len = self.max_total_len - len(prompt_ids)
            if max_resp_len <= 0:
                context_failed = True
                break

            if role == self.fixed_role and self.use_fixed_rollout:
                generate_result = await self.generate(
                    prompt_ids,
                    sampling_params={"max_new_tokens": max_resp_len},
                    rollout_name="subworker",
                )
                generate_result["logprobs"] = [0.0] * len(generate_result["output_ids"])
            else:
                generate_result = await self.generate(
                    prompt_ids,
                    sampling_params={"max_new_tokens": max_resp_len},
                )

            response_ids = generate_result["output_ids"]
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]

            response_text = self.tokenizer.decode(response_ids)

            tool_requests, tool_call_info = await self.extract_tool_calls(
                response_text, role=role
            )

            # Check whether this turn carries a final <answer> tag.
            if role in ("planner", "single"):
                has_answer, _ = self._check_final_answer(response_text)
                if has_answer:
                    has_answer_tag = True

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
                        "planner_turn_idx": planner_turn_counter,
                        "context_failed": False,
                        "max_turn_limit_failed": turn_idx >= max_turns,
                        "turn_repeat_failed": False,
                        "answer_format_failed": False,
                    },
                    tool_call_info=tool_call_info
                    if tool_call_info
                    else None,  # if passed, must have tool call
                )
            )

            prompt_ids += response_ids

            if len(response_ids) == max_resp_len:
                context_failed = True
                break

            # Determine if this is the final turn.
            # Answer tag always wins — the agent signalled completion.
            if has_answer_tag:
                succ_end = True
                if role == "planner":
                    num_turn_subagents.append(0)
                    num_effective_subagents.append(0)
                    access_search_ratio.append(0.0)
                break

            if tool_requests == []:
                # No tool calls and no answer tag → format error.
                if role in ("planner", "single"):
                    answer_format_failed = True
                else:
                    succ_end = True
                break

            # Handle tool calls based on role
            tasks = []
            tool_messages = []
            worker_buffer = []
            worker_turn_list = []
            if role == "planner":
                assert sub_traj_id == 0
                # Planner fans out multiple sub-agents in parallel.
                for i, tool_request in enumerate(tool_requests, start=1):
                    tasks.append(
                        self.worker_call(
                            tool_request,
                            origin_question,
                            is_markdown,
                            language,
                            sub_traj_id + i + sub_traj_num,
                        )
                    )
                sub_traj_num += len(tasks)
                num_turn_subagents.append(len(tasks))
                worker_results = await asyncio.gather(*tasks)

                # Count effective sub-agents and compute access/search ratio.
                num_effective = 0
                turn_total_search = 0
                turn_total_access = 0
                for worker_outputs_buffer, _, _, w_task_failed in worker_results:
                    has_search = False
                    for t in worker_outputs_buffer:
                        if t.tool_call_info:
                            s = t.tool_call_info.get("search", 0)
                            a = t.tool_call_info.get("access", 0)
                            turn_total_search += s
                            turn_total_access += a
                            if s > 0:
                                has_search = True
                    if has_search and not w_task_failed:
                        num_effective += 1
                num_effective_subagents.append(num_effective)
                ratio = (
                    turn_total_access / turn_total_search
                    if turn_total_search > 0
                    else 0.0
                )
                access_search_ratio.append(ratio)

                tool_messages_text = []
                for idx, (
                    worker_outputs_buffer,
                    worker_summary,
                    total_turn_list_worker,
                    task_failed,
                ) in enumerate(worker_results):
                    worker_buffer.extend(worker_outputs_buffer)
                    worker_turn_list.extend(total_turn_list_worker)
                    # assert len(worker_outputs_buffer) == sum(total_turn_list_worker) and len(total_turn_list_worker) >=1
                    # Format tool response with both request and result
                    subtask_text = tool_requests[idx].arguments["subtask"]
                    if not task_failed:
                        tool_messages_text.append(
                            get_planner_subtask_result_message(
                                subtask_idx=idx + 1,
                                subtask_text=subtask_text,
                                worker_summary=worker_summary,
                                language=language,
                            )
                        )
                    else:
                        tool_messages_text.append(
                            get_planner_subtask_failed_message(
                                subtask_idx=idx + 1,
                                subtask_text=subtask_text,
                                language=language,
                            )
                        )

                turn_hint = get_next_turn_hint(
                    next_turn_idx=turn_idx + 2,
                    max_turns=max_allow_turns,
                    language=language,
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": "\n\n".join(tool_messages_text) + turn_hint,
                    }
                )

            else:
                # Worker/single executes search/access tools in parallel.
                for tool_request in tool_requests:
                    tasks.append(self.tool_call(tool_request))
                tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

                tool_messages_text = []
                access_summary_jobs = []
                for idx, (tool_request, tool_response) in enumerate(
                    zip(tool_requests, tool_responses)
                ):
                    # Include the original request and the result
                    if tool_request.name == "search":
                        query = tool_request.arguments["query"]
                        tool_messages_text.append(
                            get_search_tool_message(
                                query=query,
                                search_result=tool_response.text,
                                language=language,
                            )
                        )
                    elif tool_request.name == "access":
                        url = tool_request.arguments["url"]
                        info_to_extract = tool_request.arguments["info_to_extract"]
                        page_content = tool_response.text
                        if self.use_access_summary:
                            tool_messages_text.append(None)
                            coro = self.access_sumamry(info_to_extract, page_content)
                            access_summary_jobs.append(
                                (idx, url, info_to_extract, coro)
                            )
                        else:
                            tool_messages_text.append(
                                get_access_tool_message(
                                    url=url,
                                    page_content=page_content,
                                    language=language,
                                )
                            )
                    else:
                        raise ValueError(
                            f"Unknown tool request name: {tool_request.name}"
                        )

                if self.use_access_summary and access_summary_jobs:
                    coros = [job[-1] for job in access_summary_jobs]
                    summaries = await asyncio.gather(*coros)
                    for job, summary in zip(access_summary_jobs, summaries):
                        idx, url, info_to_extract, _ = job
                        tool_messages_text[idx] = get_access_summary_tool_message(
                            url=url,
                            info_to_extract=info_to_extract,
                            summary=summary,
                            language=language,
                        )

                turn_hint = get_next_turn_hint(
                    next_turn_idx=turn_idx + 2,
                    max_turns=max_allow_turns,
                    language=language,
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": "\n\n".join(tool_messages_text) + turn_hint,
                    }
                )

            # Tokenize tool responses
            tool_response_ids = self.get_tool_response_ids(tool_messages)
            max_tool_resp_len = self.max_total_len - len(prompt_ids)
            if len(tool_response_ids) >= max_tool_resp_len:
                tool_response_failed = True
                break

            prompt_ids += tool_response_ids
            output_buffer.extend(worker_buffer)
            total_turn_list.extend(worker_turn_list)

        task_failed = self._mark_role_failed_turns(
            output_buffer=output_buffer,
            role=role,
            turn_idx=turn_idx,
            succ_end=succ_end,
            context_failed=context_failed,
            tool_response_failed=tool_response_failed,
            answer_format_failed=answer_format_failed,
        )

        # Generate summary
        if role == "planner":
            answer_text = response_text.split("<|im_end|>")[0]
        elif role == "worker":
            answer_text = (
                response_text.split("</think>")[-1].split("<|im_end|>")[0].strip()
            )
        elif role == "single":
            answer_text = response_text.split("<|im_end|>")[0]

        if role == "worker":
            total_turn_list.append(turn_idx + 1)  # with no summary
        else:
            total_turn_list.append(turn_idx + 1)
        return output_buffer, answer_text, total_turn_list, num_turn_subagents, num_effective_subagents, access_search_ratio, task_failed, succ_end

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
        language = answer.get("language", "en")
        if self.workflow == "sa":
            role = "single"
        else:
            role = "planner"

        is_markdown = answer["is_markdown"]

        (
            output_buffer,
            answer_text,
            total_turn_list,
            num_turn_subagents,
            num_effective_subagents,
            access_search_ratio,
            task_failed,
            succ_end,
        ) = await self.run_one_query_role(
            question=origin_question,
            role=role,
            sub_traj_id=sub_traj_id,
            is_markdown=is_markdown,
            language=language,
        )

        if is_markdown:
            final_answer_extract = extract_final_answer(answer_text, mode="markdown")
        else:
            final_answer_extract = extract_final_answer(answer_text, mode="boxed")

        # credit assignment
        norm_column = self.cfg.data.get("norm_column", False)
        llm_reward, format = await get_final_reward_score(
            origin_question,
            final_answer_extract,
            answer,
            is_markdown,
            norm_column,
            self.llm_generator,
        )

        reward_mode = self.cfg.agentloop.get("reward_mode", "turn")
        advantage_mode = self.cfg.algorithm.get("advantage_mode", "trajectory")

        # Optional: LLM-as-turn-RM — judge scores each planner turn.
        # Skipped when reward_mode=traj AND advantage_mode=traj, since neither
        # reward nor advantage benefits from per-turn differentiation.
        llm_turn_rewards = None
        skip_llm_judge = (
            reward_mode == "trajectory" and advantage_mode == "trajectory"
        )

        # Pre-compute outcome_reward for the LLM turn judge so it sees the
        # full traj_reward_agg (bonuses + length_penalty) rather than bare
        # llm_reward.  Mirrors the formula in credit_assignment.
        llm_as_turn_rm_cfg = self.cfg.agentloop.get("llm_as_turn_rm", False)
        if (
            llm_as_turn_rm_cfg
            and self.workflow == "mas"
            and self.llm_generator is not None
            and not skip_llm_judge
        ):
            # length_penalty
            length_limit = self.cfg.agentloop.get("length_limit", 5000)
            max_length_limit = self.cfg.agentloop.get("max_length_limit", 7000)
            length_p = self.cfg.agentloop.get("length_penalty", 0.0)
            max_response_len = max(
                (len(t.response_ids) for t in output_buffer if t.response_ids),
                default=0,
            )
            length_penalty_val = 0.0
            if max_response_len > length_limit:
                t_frac = (max_response_len - length_limit) / (max_length_limit - length_limit)
                t_frac = max(0.0, min(1.0, t_frac))
                length_penalty_val = t_frac * length_p

            p_bonus, c_bonus = _compute_traj_bonuses(
                num_turn_subagents,
                num_effective_subagents,
                output_buffer,
                self.cfg.agentloop.get("max_workers_per_planner", 10),
                self.cfg.agentloop.get("parallelism_epsilon", 0.01),
            )
            parallelism_weight = self.cfg.agentloop.get("parallelism_weight", 0.0)
            completion_weight = self.cfg.agentloop.get("completion_weight", 0.0)

            answer_format_ok = final_answer_extract is not None and format is True
            outcome_reward = (
                llm_reward
                + parallelism_weight * p_bonus
                + completion_weight * c_bonus
                - length_penalty_val
            ) if (succ_end and answer_format_ok) else 0.0

            ground_truth = answer.get("answer", "")
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0] if ground_truth else ""
            llm_turn_rewards = await evaluate_turn_rewards(
                question=origin_question,
                ground_truth=str(ground_truth),
                outcome_reward=outcome_reward,
                output_buffer=output_buffer,
                judge_llm_generator=self.llm_generator,
                num_planner_turns=len(num_turn_subagents)
                if num_turn_subagents
                else 0,
            )

        # Hindsight importance weights (reward_mode=trajectory, advantage_mode=turn,
        # llm_as_turn_rm=False).  Scores each planner turn by how necessary it was
        # for the final outcome, then weights turn rewards by ρ×γ^(T-t).
        hind_weights = None
        llm_as_turn_rm = self.cfg.agentloop.get("llm_as_turn_rm", False)
        if (
            reward_mode == "trajectory"
            and advantage_mode == "turn"
            and not llm_as_turn_rm
            and self.workflow == "mas"
        ):
            hind_weights = await compute_hind_weights(
                output_buffer=output_buffer,
                extract_answer=str(final_answer_extract),
                temperature=self.cfg.agentloop.get("hind_temperature", 1.0),
                c_min=self.cfg.agentloop.get("hind_clip_min", 0.1),
                c_max=self.cfg.agentloop.get("hind_clip_max", 5.0),
                compute_logprobs=self.compute_token_logprobs,
            )

        output_buffer, train_buffer, final_answer_format, turn_rewards = (
            credit_assignment(
                agentloop_config=self.cfg.agentloop,
                output_buffer=output_buffer,
                llm_reward=llm_reward,
                succ_end=succ_end,
                answer_format=final_answer_extract is not None and format is True,
                num_turn_subagents=num_turn_subagents
                if self.workflow == "mas"
                else None,
                num_effective_subagents=num_effective_subagents
                if self.workflow == "mas"
                else None,
                access_search_ratio=access_search_ratio
                if self.workflow == "mas"
                else None,
                llm_turn_rewards=llm_turn_rewards,
                hind_weights=hind_weights,
            )
        )

        assert len(turn_rewards) == len(output_buffer), (
            f"turn_rewards lens mismatch: {len(turn_rewards)} != {len(output_buffer)}"
        )
        for i, single_turn_output in enumerate(output_buffer):
            single_turn_output.reward_score = turn_rewards[i]
        for single_turn_output in train_buffer:
            idx = output_buffer.index(single_turn_output)
            single_turn_output.reward_score = turn_rewards[idx]

        for single_turn_output in output_buffer:
            single_turn_output.extra_fields["not_training"] = (
                False if self.is_eval else True
            )
        for single_turn_output in train_buffer:
            single_turn_output.extra_fields["not_training"] = False

        # Track valid turns for computing averages
        num_valid_planner_turns = 0
        num_valid_worker_turns = 0

        for single_turn_output in output_buffer:
            # Collect tool call info (keep all turns but track valid ones)
            single_turn_output: AgentLoopOutput
            subtask_count = 0
            search_count = 0
            access_count = 0
            if single_turn_output.tool_call_info is not None:
                role = single_turn_output.tool_call_info.get("role", "")
                subtask_count = single_turn_output.tool_call_info.get("subtask", 0)
                search_count = single_turn_output.tool_call_info.get("search", 0)
                access_count = single_turn_output.tool_call_info.get("access", 0)

                # Track valid turns by role
                if role == "planner":
                    assert subtask_count > 0
                    num_valid_planner_turns += 1
                elif role == "worker" or role == "single":
                    assert search_count > 0 or access_count > 0
                    num_valid_worker_turns += 1
            single_turn_output.extra_fields["subtask_count"] = subtask_count
            single_turn_output.extra_fields["search_count"] = search_count
            single_turn_output.extra_fields["access_count"] = access_count
            single_turn_output.extra_fields["tool_call_info"] = (
                single_turn_output.tool_call_info
            )
            single_turn_output.extra_fields["prompt_text"] = (
                single_turn_output.prompt_text
            )
            single_turn_output.extra_fields["response_text"] = (
                single_turn_output.response_text
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
                "total_turn_list": total_turn_list if self.workflow == "mas" else None,
                "num_turn_subagents": num_turn_subagents if self.workflow == "mas" else None,
                "num_effective_subagents": num_effective_subagents if self.workflow == "mas" else None,
                "access_search_ratio": access_search_ratio if self.workflow == "mas" else None,
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
        planner_turn_idx_list = []
        for task_result in task_results:
            sub_traj_map = {}
            for single_turn_output in task_result.single_turn_outputs:
                if single_turn_output.extra_fields["not_training"]:
                    continue
                role_idx = single_turn_output.extra_fields["idx_to_sub_traj"]
                if role_idx not in sub_traj_map:
                    sub_traj_map[role_idx] = len(sub_traj_map)
                idx_to_sub_traj.append(sub_traj_map[role_idx])
                planner_turn_idx_list.append(
                    single_turn_output.extra_fields.get("planner_turn_idx", -1)
                )
        extra_fields_train = {
            "idx_to_sub_traj": idx_to_sub_traj,
            "planner_turn_idx": planner_turn_idx_list,
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
        if self.is_eval:
            return {}

        rollout_batch = {
            "turn_subtask_counts": rollout_result.extra_fields_turn["subtask_count"],
            "turn_search_counts": rollout_result.extra_fields_turn["search_count"],
            "turn_access_counts": rollout_result.extra_fields_turn["access_count"],
            "num_valid_planner_turns": sum(
                rollout_result.extra_fields_traj["num_valid_planner_turns"]
            ),
            "num_valid_worker_turns": sum(
                rollout_result.extra_fields_traj["num_valid_worker_turns"]
            ),
            "total_turn_list_metric": rollout_result.extra_fields_traj[
                "total_turn_list"
            ],
            "num_turn_subagents_metric": rollout_result.extra_fields_traj[
                "num_turn_subagents"
            ],
            "num_effective_subagents_metric": rollout_result.extra_fields_traj[
                "num_effective_subagents"
            ],
            "access_search_ratio_metric": rollout_result.extra_fields_traj[
                "access_search_ratio"
            ],
            "final_answer_format": rollout_result.extra_fields_traj[
                "final_answer_format"
            ],
        }
        return _compute_rollout_metrics(
            rollout_batch=rollout_batch,
            idx_to_traj=rollout_result.idx_to_traj,
            num_trajectories=int(rollout_result.group_size),
        )
