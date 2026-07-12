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
from typing import Any

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import ToolResponse
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
        max_total_len = int(self.cfg.runner.seq_length)
        self.max_resp_len = max(1, max_total_len - self.max_prompt_len)
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

    async def pre_process_query(
        self, prompt_ids: list[int], answer: str
    ) -> tuple[list[int], dict[str, Any]]:
        """Prepare a query using an opaque reward-reference ID, never GT."""
        return (
            prompt_ids[: self.max_prompt_len],
            {
                "reference_id": answer,
                "next_turn_id": 0,
                "all_llm_response_ids": [],  # accumulate only LLM-generated tokens for reward
                "problem_prompt_ids": copy.deepcopy(prompt_ids[: self.max_prompt_len]),
                "last_llm_output": None,
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

        output.extra_fields["llm_reward"] = 0.0
        output.extra_fields["response_text"] = final_response_text
        output.extra_fields["prompt_text"] = self.tokenizer.decode(
            generate_context.get("problem_prompt_ids", [])
        )
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
        max_resp_len = self.max_resp_len - (
            len(turn_prompt_ids) - len(problem_prompt_ids)
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
        available_tool_tokens = self.max_resp_len - (
            len(turn_prompt_ids) + len(llm_response_ids) - len(problem_prompt_ids)
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
        next_turn_prompt_ids = turn_prompt_ids + llm_response_ids + tool_response_ids
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
            "format_valid": [],
            "prompt_text": [],
            "response_text": [],
        }
        extra_fields_traj = {
            "llm_reward": [],
            "response_text": [],
            "prompt_text": [],
            "turns": [],
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
            for single_turn_output in task_result.single_turn_outputs:
                metadata = single_turn_output.extra_fields
                for key in (
                    "turn_id",
                    "is_search",
                    "is_terminal",
                    "search_query",
                    "visible_evidence",
                    "format_valid",
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

        return (
            extra_fields_turn,
            extra_fields_traj,
            {"reference_id": answer},
            extra_fields_train,
        )
