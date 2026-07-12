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

import asyncio

import torch
from omegaconf import OmegaConf

from rlinf.agents.searchr1.reference_runner import SearchR1ReferenceRunnerMixin
from rlinf.agents.searchr1.reward_worker import assign_searchr1_rewards
from rlinf.agents.searchr1.searchr1_agent_loop import (
    Searchr1AgentLoopWorker,
    truncate_token_ids,
)
from rlinf.algorithms.advantages import compute_grpo_dynamic_advantages
from rlinf.data.io_struct import DynamicRolloutResult
from rlinf.data.tool_call.tool_io_struct import ToolRequest, ToolResponse
from rlinf.workers.agent.agent_loop import AgentLoopOutput, MultiAgentLoopOutput


class _CaptureChannel:
    def __init__(self):
        self.items = []

    def put(self, value, async_op=False):
        self.items.append(value)


class _DummyReferenceRunner(SearchR1ReferenceRunnerMixin):
    pass


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


def _turn(turn_id, *, is_search, is_terminal):
    return AgentLoopOutput(
        prompt_ids=[turn_id],
        response_ids=[turn_id + 10],
        prompt_text=f"prompt-{turn_id}",
        response_text=f"response-{turn_id}",
        is_end=is_terminal,
        reward_score=0.0,
        extra_fields={
            "turn_id": turn_id,
            "is_search": is_search,
            "is_terminal": is_terminal,
            "search_query": "query" if is_search else None,
            "visible_evidence": "evidence" if is_search else None,
            "format_valid": True,
        },
    )


def test_reference_runner_keeps_ground_truth_out_of_rollout_request():
    runner = _DummyReferenceRunner()
    runner.cfg = OmegaConf.create({"algorithm": {"group_size": 2}})
    runner.component_placement = type("Placement", (), {"rollout_dp_size": 1})()
    runner.dataloader_channel = _CaptureChannel()
    runner.reward_reference_channel = _CaptureChannel()
    batch = {
        "prompt": torch.tensor([[0, 11, 12]]),
        "length": torch.tensor([2]),
        "answer": [["secret-ground-truth"]],
        "image_data": [None],
        "multi_modal_inputs": [None],
    }

    runner._put_batch(batch)

    rollout_request = runner.dataloader_channel.items[0]
    reference_batch = runner.reward_reference_channel.items[0]
    assert rollout_request.input_ids == [[11, 12]]
    assert rollout_request.answers == reference_batch["reference_ids"]
    assert "secret-ground-truth" not in repr(rollout_request)
    assert reference_batch["answers"] == [["secret-ground-truth"]]


def test_preprocess_stores_only_opaque_reference_id():
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.max_prompt_len = 2

    prompt_ids, context = asyncio.run(
        worker.pre_process_query([1, 2, 3], "opaque-reference-id")
    )

    assert prompt_ids == [1, 2]
    assert context["reference_id"] == "opaque-reference-id"
    assert "answer" not in context


def test_gen_extra_fields_emits_turn_and_terminal_training_metadata():
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    trajectory = MultiAgentLoopOutput(
        single_turn_outputs=[
            _turn(0, is_search=True, is_terminal=False),
            _turn(1, is_search=False, is_terminal=True),
        ],
        extra_fields={
            "llm_reward": 0.0,
            "response_text": "full response",
            "prompt_text": "question",
            "turns": [],
        },
    )

    turn_fields, trajectory_fields, group_fields, train_fields = (
        worker.gen_extra_fields([trajectory], "opaque-reference-id")
    )

    assert turn_fields["turn_id"] == [0, 1]
    assert turn_fields["is_search"] == [True, False]
    assert train_fields["planner_turn_idx"] == [0, 1]
    assert train_fields["is_terminal"] == [False, True]
    assert trajectory_fields["response_text"] == ["full response"]
    assert group_fields == {"reference_id": "opaque-reference-id"}


def test_assign_reward_only_to_each_trajectory_terminal_turn():
    rollout_result = DynamicRolloutResult(
        num_sequence=3,
        group_size=2,
        idx_to_traj=[0, 0, 1],
        input_ids=[[1], [2], [3]],
        prompt_lengths=[0, 0, 0],
        response_lengths=[1, 1, 1],
        is_end=[False, True, True],
        rewards=[0.0, 0.0, 0.0],
        extra_fields_train={
            "planner_turn_idx": [0, 1, 0],
            "is_terminal": [False, True, True],
        },
        extra_fields_traj={
            "response_text": [
                "<search>capital of France</search><answer>Paris</answer>",
                "<answer>London</answer>",
            ]
        },
        extra_fields_group={"reference_id": "opaque-reference-id"},
    )

    scores = assign_searchr1_rewards(rollout_result, ["Paris"])

    assert scores == [1.0, 0.0]
    assert rollout_result.rewards == [0.0, 1.0, 0.0]
    assert rollout_result.extra_fields_traj["llm_reward"] == [1.0, 0.0]
    assert "answer" not in rollout_result.extra_fields_group

    assign_searchr1_rewards(rollout_result, ["Paris"], expose_reference=True)
    assert rollout_result.extra_fields_group["answer"] == ["Paris"]


def test_terminal_only_reward_produces_one_trajectory_advantage():
    rewards = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    advantages, _ = compute_grpo_dynamic_advantages(
        rewards=rewards,
        loss_mask=torch.ones(3, 4),
        group_size=2,
        idx_to_traj=[0, 0, 1, 1],
        planner_turn_idx=[0, 1, 0, 1],
        advantage_mode="trajectory",
        reward_mode="trajectory",
    )

    assert torch.equal(advantages[:, 0], advantages[:, 1])
    assert torch.equal(advantages[:, 2], advantages[:, 3])
    assert torch.all(advantages[:, 0] > 0)
    assert torch.all(advantages[:, 2] < 0)


def test_tool_response_truncation_uses_token_count_and_requested_side():
    token_ids = list(range(10))

    assert truncate_token_ids(token_ids, 4, "right") == [0, 1, 2, 3]
    assert truncate_token_ids(token_ids, 4, "left") == [6, 7, 8, 9]
    assert truncate_token_ids(token_ids, 5, "middle") == [0, 1, 7, 8, 9]


def test_generate_tool_response_records_exact_policy_visible_evidence():
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.cfg = OmegaConf.create({"agentloop": {"max_turns": 2}})
    worker.max_resp_len = 20
    worker.max_tool_response_length = 4
    worker.tool_response_truncate_side = "right"
    worker.print_outputs = False
    worker.tokenizer = _CharacterTokenizer()

    async def parse_tool_call(_text):
        return "", [
            ToolRequest(name="search", arguments={"keyword": "capital of France"})
        ]

    async def call_tool(_request):
        return ToolResponse(text="abcdefghij")

    worker.toolcall_parser = parse_tool_call
    worker.tool_call = call_tool
    llm_output = _turn(0, is_search=False, is_terminal=True)
    context = {"next_turn_id": 1, "last_llm_output": llm_output}

    is_continue, next_prompt = asyncio.run(
        worker.generate_tool_response(
            context,
            [],
            [1, 2],
            [1, 2],
            [3],
            "<search>capital of France</search>",
        )
    )

    assert is_continue is True
    assert next_prompt[-4:] == [ord(character) for character in "abcd"]
    assert llm_output.extra_fields["visible_evidence"] == "abcd"
    assert llm_output.extra_fields["is_search"] is True
    assert llm_output.extra_fields["is_terminal"] is False


def _run_scripted_trajectory(max_turns, responses):
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.cfg = OmegaConf.create({"agentloop": {"max_turns": max_turns}})
    worker.max_prompt_len = 32
    worker.max_resp_len = 200
    worker.max_tool_response_length = 4
    worker.tool_response_truncate_side = "right"
    worker.print_outputs = False
    worker.return_logprobs = False
    worker.tokenizer = _CharacterTokenizer()
    generated_texts = iter(responses)

    async def generate(_prompt_ids, sampling_params=None):
        del sampling_params
        return {"output_ids": worker.tokenizer.encode(next(generated_texts))}

    async def parse_tool_call(text):
        if "<search>" not in text:
            return text, []
        return "", [ToolRequest(name="search", arguments={"keyword": "capital France"})]

    async def call_tool(_request):
        return ToolResponse(text="Paris is the capital of France")

    worker.generate = generate
    worker.toolcall_parser = parse_tool_call
    worker.tool_call = call_tool
    return worker, asyncio.run(
        worker.run_one_query(worker.tokenizer.encode("question"), answer="opaque-id")
    )


def test_one_search_then_answer_runs_end_to_end_with_terminal_reward():
    worker, trajectory = _run_scripted_trajectory(
        2, ["<search>capital France</search>", "<answer>Paris</answer>"]
    )

    assert len(trajectory.single_turn_outputs) == 2
    assert trajectory.single_turn_outputs[0].extra_fields["is_search"] is True
    assert trajectory.single_turn_outputs[0].extra_fields["is_terminal"] is False
    assert trajectory.single_turn_outputs[1].extra_fields["is_terminal"] is True
    extra_fields = worker.gen_extra_fields([trajectory], "opaque-id")
    rollout_result = worker.get_rollout_result([trajectory], *extra_fields)

    assign_searchr1_rewards(rollout_result, ["Paris"])

    assert rollout_result.rewards == [0.0, 1.0]
    assert rollout_result.extra_fields_group == {"reference_id": "opaque-id"}


def test_max_turns_three_allows_two_searches_then_an_answer():
    worker, trajectory = _run_scripted_trajectory(
        3,
        [
            "<search>France capital</search>",
            "<search>Paris city</search>",
            "<answer>Paris</answer>",
        ],
    )

    assert len(trajectory.single_turn_outputs) == 3
    assert [
        output.extra_fields["is_search"] for output in trajectory.single_turn_outputs
    ] == [True, True, False]
    assert [
        output.extra_fields["is_terminal"] for output in trajectory.single_turn_outputs
    ] == [False, False, True]
    rollout_result = worker.get_rollout_result(
        [trajectory], *worker.gen_extra_fields([trajectory], "opaque-id")
    )

    assign_searchr1_rewards(rollout_result, ["Paris"])

    assert rollout_result.rewards == [0.0, 0.0, 1.0]
