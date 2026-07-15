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
import json
from collections import defaultdict

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.agents.searchr1.reference_runner import SearchR1ReferenceRunnerMixin
from rlinf.agents.searchr1.reward_worker import assign_searchr1_rewards
from rlinf.agents.searchr1.searchr1_agent_loop import (
    Searchr1AgentLoopWorker,
    truncate_token_ids,
)
from rlinf.agents.searchr1.teacher_planner import (
    FrozenTeacherPlanner,
    TeacherPlan,
    TeacherPlanResult,
    build_guidance_token_ids,
    build_shadow_metrics,
    paired_bootstrap_ci,
    parse_teacher_plan,
    shuffled_teacher_plans,
    teacher_plan_cache_key,
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
    assert group_fields["reference_id"] == "opaque-reference-id"
    assert "answer" not in group_fields


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
    assert rollout_result.extra_fields_group["reference_id"] == "opaque-id"
    assert "answer" not in rollout_result.extra_fields_group


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


def test_teacher_plan_parser_enforces_exact_compact_schema():
    response = json.dumps(
        {
            "goal": "identify the relevant entity",
            "query_intent": "find a primary source",
            "expected_evidence": "an explicit entity-relation statement",
            "fallback": "search the related organization",
        }
    )

    plan = parse_teacher_plan(response)

    assert plan.goal == "identify the relevant entity"
    with pytest.raises(ValueError, match="only one JSON object"):
        parse_teacher_plan(f"Here is the plan: {response}")
    with pytest.raises(ValueError, match="fields must be exactly"):
        parse_teacher_plan(response[:-1] + ', "answer": "secret"}')
    with pytest.raises(ValueError, match="empty or too long"):
        parse_teacher_plan(response.replace("find a primary source", ""))


def test_teacher_plan_cache_key_is_versioned_and_deterministic():
    first = teacher_plan_cache_key("question", "teacher-v1", 1234)

    assert first == teacher_plan_cache_key("question", "teacher-v1", 1234)
    assert first != teacher_plan_cache_key("question", "teacher-v2", 1234)
    assert first != teacher_plan_cache_key("question", "teacher-v1", 2025)


class _TeacherTokenizer(_CharacterTokenizer):
    def __init__(self):
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        self.last_messages = messages
        return self.encode(json.dumps(messages))

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return super().decode(token_ids)


def test_frozen_teacher_receives_no_gt_and_reuses_cache(tmp_path):
    tokenizer = _TeacherTokenizer()
    cfg = OmegaConf.create(
        {
            "data": {"seed": 1234},
            "teacher_planner": {
                "version": "teacher-v1",
                "seed": 1234,
                "cache_dir": str(tmp_path),
                "max_new_tokens": 64,
            },
        }
    )
    planner = FrozenTeacherPlanner(cfg, tokenizer)
    response = json.dumps(
        {
            "goal": "locate evidence",
            "query_intent": "search the named subject",
            "expected_evidence": "a direct statement",
            "fallback": "search a related source",
        }
    )
    generate_calls = []

    async def generate(prompt_ids, sampling_params, rollout_name):
        generate_calls.append((prompt_ids, sampling_params, rollout_name))
        assert "secret-ground-truth" not in tokenizer.decode(prompt_ids)
        return {"output_ids": tokenizer.encode(response)}

    first = asyncio.run(planner.get_plan("public question", generate))
    second = asyncio.run(planner.get_plan("public question", generate))

    assert first.valid is True
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(generate_calls) == 1
    assert tokenizer.last_messages[-1] == {
        "role": "user",
        "content": "public question",
    }


def test_generic_guidance_matches_real_guidance_token_length():
    tokenizer = _CharacterTokenizer()
    plan = TeacherPlan(
        goal="specific goal",
        query_intent="specific query",
        expected_evidence="specific evidence",
        fallback="specific fallback",
    )

    guided_ids = build_guidance_token_ids(tokenizer, plan, "guided")
    generic_ids = build_guidance_token_ids(tokenizer, plan, "generic")

    assert len(generic_ids) == len(guided_ids)
    assert generic_ids != guided_ids


def test_shuffled_teacher_control_is_reproducible_derangement():
    plans = [
        TeacherPlanResult(
            plan_id=str(index),
            valid=True,
            plan=TeacherPlan("goal", "intent", "evidence", "fallback"),
            raw_response="{}",
        )
        for index in range(5)
    ]

    shuffled = shuffled_teacher_plans(plans, list(range(5)), seed=1234)

    assert shuffled == shuffled_teacher_plans(plans, list(range(5)), seed=1234)
    assert all(
        original.plan_id != control.plan_id
        for original, control in zip(plans, shuffled)
    )


def test_paired_bootstrap_ci_is_deterministic_and_positive():
    first = paired_bootstrap_ci([0.2, 0.4, 0.3, 0.5], seed=1234, num_samples=200)
    second = paired_bootstrap_ci([0.2, 0.4, 0.3, 0.5], seed=1234, num_samples=200)

    assert first == second
    assert first[0] > 0
    assert first[1] >= first[0]


def test_teacher_guidance_is_removed_after_first_search():
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.cfg = OmegaConf.create(
        {
            "agentloop": {"max_turns": 2},
            "rollout": {"model": {"model_path": "policy"}},
        }
    )
    worker.max_prompt_len = 32
    worker.max_resp_len = 64
    worker.max_total_len = 128
    worker.max_tool_response_length = 8
    worker.tool_response_truncate_side = "right"
    worker.print_outputs = False
    worker.tokenizer = _CharacterTokenizer()
    worker.teacher_planner = type("Teacher", (), {"teacher_version": "teacher-v1"})()
    plan_result = TeacherPlanResult(
        plan_id="plan-id",
        valid=True,
        plan=TeacherPlan("goal", "intent", "evidence", "fallback"),
        raw_response="{}",
    )
    original_prompt = worker.tokenizer.encode("question")
    guided_prompt, context = asyncio.run(
        worker.pre_process_query(
            original_prompt,
            "opaque-id",
            question_text="question",
            sample_id=7,
            guidance_mode="guided",
            teacher_plan_result=plan_result,
        )
    )

    async def parse_tool_call(_text):
        return "", [ToolRequest(name="search", arguments={"keyword": "query"})]

    async def call_tool(_request):
        return ToolResponse(text="evidence")

    worker.toolcall_parser = parse_tool_call
    worker.tool_call = call_tool
    llm_output = _turn(0, is_search=False, is_terminal=True)
    context["last_llm_output"] = llm_output
    context["next_turn_id"] = 1
    is_continue, next_prompt = asyncio.run(
        worker.generate_tool_response(
            context,
            [],
            guided_prompt,
            guided_prompt,
            worker.tokenizer.encode("<search>query</search>"),
            "<search>query</search>",
        )
    )

    assert is_continue is True
    assert guided_prompt != original_prompt
    assert next_prompt[: len(original_prompt)] == original_prompt
    assert "teacher_guidance" not in worker.tokenizer.decode(next_prompt)


def test_shadow_metric_names_include_paired_uplift_and_controls():
    context = {
        "mode_reward_sums": defaultdict(
            float, {"guided": 3.0, "unguided": 1.0, "generic": 1.0}
        ),
        "mode_answer_hit_sums": defaultdict(
            float, {"guided": 4.0, "unguided": 2.0, "generic": 2.0}
        ),
        "mode_counts": defaultdict(int, {"guided": 4, "unguided": 4, "generic": 4}),
        "paired_uplift_sums": defaultdict(float, {"guided": 1.0, "generic": 0.0}),
        "paired_uplifts": defaultdict(
            list, {"guided": [0.5, 0.5], "generic": [0.0, 0.0]}
        ),
        "paired_answer_hit_sums": defaultdict(float, {"guided": 1.0, "generic": 0.0}),
        "paired_counts": defaultdict(int, {"guided": 2, "generic": 2}),
        "query_change_sums": defaultdict(float, {"guided": 2.0, "generic": 1.0}),
        "query_change_counts": defaultdict(int, {"guided": 2, "generic": 2}),
        "plan_valid_by_id": {"a": True, "b": False},
        "plan_cache_hit_by_id": {"a": True, "b": False},
    }

    metrics = build_shadow_metrics(context, bootstrap_samples=200)

    assert metrics["eval/unguided_EM"] == 0.25
    assert metrics["planner/guided_EM"] == 0.75
    assert metrics["planner/guided_minus_unguided"] == 0.5
    assert metrics["planner/plan_valid_rate"] == 0.5
    assert metrics["planner/query_change_rate"] == 1.0
    assert metrics["planner/answer_hit_delta"] == 0.5
    assert metrics["planner/generic_minus_unguided"] == 0.0
    assert metrics["planner/guided_uplift_ci_low"] == 0.5
    assert metrics["planner/guided_uplift_ci_high"] == 0.5
