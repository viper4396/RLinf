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
import re
from collections import defaultdict

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.agents.searchr1.eval_diagnostics import (
    audit_plan_semantic_coverage,
    build_abc_acceptance_metrics,
    build_label_only_diagnostics,
)
from rlinf.agents.searchr1.reference_runner import SearchR1ReferenceRunnerMixin
from rlinf.agents.searchr1.reward_worker import assign_searchr1_rewards
from rlinf.agents.searchr1.searchr1_agent_loop import (
    Searchr1AgentLoopWorker,
    classify_controller_binding_failure,
    controller_fallback_query,
    extract_controller_bound_query,
    extract_controller_evidence_titles,
    merge_search_response_ids,
    normalize_controller_synthesis_response,
    truncate_token_ids,
)
from rlinf.agents.searchr1.teacher_planner import (
    FrozenTeacherPlanner,
    TeacherPlan,
    TeacherPlanResult,
    TeacherPlanStep,
    build_guidance_token_ids,
    build_shadow_metrics,
    extract_query_anchors,
    extract_searchr1_question,
    insert_guidance_user_message,
    load_teacher_questions,
    paired_bootstrap_ci,
    parse_teacher_plan,
    shuffled_teacher_plans,
    teacher_plan_cache_key,
    validate_teacher_plan_semantics,
)
from rlinf.algorithms.advantages import compute_grpo_dynamic_advantages
from rlinf.algorithms.toolcall_parsers import Searchr1QwenToolCallParser
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


def _multihop_plan() -> TeacherPlan:
    return TeacherPlan(
        decision="PLAN",
        plan_type="sequential",
        steps=(
            TeacherPlanStep(
                step_id=1,
                goal="Resolve the intermediate entity",
                query_template="source entity relation",
                expected_evidence="the intermediate entity name",
                depends_on=(),
            ),
            TeacherPlanStep(
                step_id=2,
                goal="Retrieve the requested fact",
                query_template="{step_1_result} target relation",
                expected_evidence="the final supporting fact",
                depends_on=(1,),
            ),
        ),
    )


def _comparison_plan() -> TeacherPlan:
    return TeacherPlan(
        decision="PLAN",
        plan_type="comparison",
        steps=(
            TeacherPlanStep(
                step_id=1,
                goal="Find Alpha's release year",
                query_template="Alpha release year",
                expected_evidence="Alpha release year",
                depends_on=(),
            ),
            TeacherPlanStep(
                step_id=2,
                goal="Find Beta's release year",
                query_template="Beta release year",
                expected_evidence="Beta release year",
                depends_on=(),
            ),
        ),
    )


def _controller_worker(max_turns=3):
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.cfg = OmegaConf.create({"agentloop": {"max_turns": max_turns}})
    worker.max_prompt_len = 1024
    worker.max_total_len = 4096
    worker.max_resp_len = 2048
    worker.max_tool_response_length = 512
    worker.tool_response_truncate_side = "right"
    worker.print_outputs = False
    worker.tokenizer = _CharacterTokenizer()
    worker.teacher_planner = type("Teacher", (), {"teacher_version": "teacher-v3"})()
    worker.teacher_execution_mode = "controller"
    worker.controller_max_evidence_length = 2048
    worker.controller_min_synthesis_tokens = 128
    worker.dual_query_retrieval = False
    worker.use_fallback_query = False
    worker.persist_teacher_plan = False
    worker.force_search_on_first_turn = True
    return worker


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


def test_merge_search_responses_splits_one_total_budget_between_sources():
    tokenizer = _CharacterTokenizer()
    merged_ids = merge_search_response_ids(
        tokenizer,
        [
            ToolResponse(
                text="<information>" + "a" * 100 + "</information>\n<think>\n"
            ),
            ToolResponse(
                text="<information>" + "b" * 100 + "</information>\n<think>\n"
            ),
        ],
        ["original question", "teacher supplement"],
        max_length=160,
        truncate_side="right",
    )
    merged_text = tokenizer.decode(merged_ids)

    assert len(merged_ids) == 160
    assert "Search results: original question" in merged_text
    assert "Search results: teacher supplement" in merged_text
    assert "a" in merged_text and "b" in merged_text
    assert merged_text.endswith("\n<think>\n")


def test_searchr1_parser_repairs_common_search_tag_errors():
    parser = Searchr1QwenToolCallParser()

    _, malformed_open = asyncio.run(parser("<search query>capital of France</search>"))
    _, missing_close = asyncio.run(parser("<search>capital of France"))

    assert malformed_open[0].arguments["keyword"] == "capital of France"
    assert missing_close[0].arguments["keyword"] == "capital of France"


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


def test_guided_multihop_plan_executes_policy_queries_sequentially():
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.cfg = OmegaConf.create({"agentloop": {"max_turns": 3}})
    worker.max_resp_len = 200
    worker.max_total_len = 2000
    worker.max_tool_response_length = 500
    worker.tool_response_truncate_side = "right"
    worker.print_outputs = False
    worker.tokenizer = _CharacterTokenizer()
    worker.dual_query_retrieval = True
    worker.use_fallback_query = True
    worker.persist_teacher_plan = True
    worker.force_search_on_first_turn = True
    executed_queries = []

    async def parse_tool_call(_text):
        return "", [ToolRequest(name="search", arguments={"keyword": "policy query"})]

    async def call_tool(request):
        query = request.arguments["keyword"]
        executed_queries.append(query)
        return ToolResponse(text=f"<information>{query}</information>\n<think>\n")

    worker.toolcall_parser = parse_tool_call
    worker.tool_call = call_tool
    llm_output = _turn(0, is_search=False, is_terminal=True)
    context = {
        "next_turn_id": 1,
        "last_llm_output": llm_output,
        "question_text": "original question",
        "teacher_rewrite_applied": True,
        "teacher_plan_type": "sequential",
        "teacher_plan_id": "plan-id",
        "teacher_plan_step_count": 2,
        "teacher_plan_search_count": 0,
        "teacher_supplemental_query": "safe supplemental query",
        "teacher_fallback_query": "safe fallback query",
        "guidance_applied": False,
    }

    is_continue, _ = asyncio.run(
        worker.generate_tool_response(
            context,
            [],
            [1],
            [1],
            worker.tokenizer.encode("<search>policy query</search>"),
            "<search>policy query</search>",
        )
    )

    assert is_continue is True
    assert executed_queries == ["policy query"]
    assert llm_output.extra_fields["executed_search_queries"] == executed_queries
    assert llm_output.extra_fields["dual_query_applied"] is False
    assert llm_output.extra_fields["teacher_plan_step_id"] == 1
    assert llm_output.extra_fields["teacher_plan_node_id"] == "plan-id:hop_1"

    executed_queries.clear()
    second_llm_output = _turn(1, is_search=False, is_terminal=True)
    context["last_llm_output"] = second_llm_output
    context["next_turn_id"] = 2
    is_continue, _ = asyncio.run(
        worker.generate_tool_response(
            context,
            [],
            [1],
            [1],
            worker.tokenizer.encode("<search>policy second query</search>"),
            "<search>policy second query</search>",
        )
    )

    assert is_continue is True
    assert executed_queries == ["policy query"]
    assert second_llm_output.extra_fields["dual_query_applied"] is False
    assert second_llm_output.extra_fields["teacher_plan_step_id"] == 2
    assert second_llm_output.extra_fields["teacher_plan_node_id"] == "plan-id:hop_2"


def test_controller_preprocess_exposes_only_current_hop():
    worker = _controller_worker()
    prompt_ids = worker.tokenizer.encode(
        "<|im_start|>user\nQuestion: target?<|im_end|>\n<|im_start|>assistant\n"
    )
    result = TeacherPlanResult(
        plan_id="plan-id",
        valid=True,
        plan=_multihop_plan(),
        raw_response="{}",
    )

    controlled_ids, context = asyncio.run(
        worker.pre_process_query(
            prompt_ids,
            "opaque-id",
            question_text="target?",
            sample_id=7,
            guidance_mode="guided",
            teacher_plan_result=result,
        )
    )
    controlled_text = worker.tokenizer.decode(controlled_ids)

    assert context["teacher_controller_applied"] is True
    assert context["guidance_applied"] is True
    assert context["controller_phase"] == "hop"
    assert "CONTROLLED SEARCH HOP" in controlled_text
    assert "source entity relation" in controlled_text
    assert "target relation" not in controlled_text
    assert "UNTRUSTED SEARCH PLAN" not in controlled_text


def test_controller_executes_sequential_hops_then_isolated_synthesis():
    worker = _controller_worker()
    prompt_ids = worker.tokenizer.encode(
        "<|im_start|>user\nQuestion: Where did the parent die?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    result = TeacherPlanResult(
        plan_id="plan-id",
        valid=True,
        plan=_multihop_plan(),
        raw_response="{}",
    )
    executed_queries = []
    generated_prompts = []

    async def parse_tool_call(text):
        match = re.search(r"<search>\s*(.*?)\s*</search>", text, re.DOTALL)
        requests = (
            [ToolRequest(name="search", arguments={"keyword": match.group(1)})]
            if match
            else []
        )
        return "", requests

    async def call_tool(request):
        query = request.arguments["keyword"]
        executed_queries.append(query)
        evidence = (
            "<information>The source entity is Parent Entity.</information>"
            if query == "source entity relation"
            else "<information>Parent Entity died in Target City.</information>"
        )
        return ToolResponse(text=evidence)

    async def generate(prompt, sampling_params):
        del sampling_params
        prompt_text = worker.tokenizer.decode(prompt)
        generated_prompts.append(prompt_text)
        if "CONTROLLED SYNTHESIS" in prompt_text:
            assert prompt_text.endswith("<think>")
            text = "The evidence names the city.</think><answer>Target City</answer>"
        else:
            assert prompt_text.endswith('{"resolved_values":')
            assert "Current query template: {step_1_result} target relation" in (
                prompt_text
            )
            assert "Parent Entity" in prompt_text
            text = (
                '{"step_1_result":"Parent Entity"},'
                '"query":"Parent Entity target relation"}'
            )
        return {"output_ids": worker.tokenizer.encode(text)}

    worker.toolcall_parser = parse_tool_call
    worker.tool_call = call_tool
    worker.generate = generate

    output = asyncio.run(
        worker.run_one_query(
            prompt_ids,
            answer="opaque-id",
            question_text="Where did the parent die?",
            sample_id=7,
            guidance_mode="guided",
            teacher_plan_result=result,
        )
    )

    assert executed_queries == [
        "source entity relation",
        "Parent Entity target relation",
    ]
    assert [
        turn.extra_fields["controller_phase"] for turn in output.single_turn_outputs
    ] == [
        "hop",
        "hop",
        "synthesis",
    ]
    assert output.single_turn_outputs[0].extra_fields["not_training"] is True
    assert output.single_turn_outputs[0].extra_fields["controller_query_source"] == (
        "template"
    )
    assert output.single_turn_outputs[1].extra_fields["controller_query_source"] == (
        "policy"
    )
    dependent_turn = output.single_turn_outputs[1]
    synthesis_turn = output.single_turn_outputs[2]
    assert worker.tokenizer.decode(dependent_turn.prompt_ids).endswith(
        '{"resolved_values":'
    )
    assert worker.tokenizer.decode(dependent_turn.response_ids) == (
        '{"step_1_result":"Parent Entity"},"query":"Parent Entity target relation"}'
    )
    assert worker.tokenizer.decode(synthesis_turn.prompt_ids).endswith("<think>")
    assert worker.tokenizer.decode(synthesis_turn.response_ids) == (
        "The evidence names the city.</think><answer>Target City</answer>"
    )
    assert output.extra_fields["controller_completed_step_ids"] == [1, 2]
    assert output.extra_fields["controller_completed"] is True
    assert output.extra_fields["controller_synthesis_generated"] is True
    assert output.extra_fields["controller_fallback_query_count"] == 0
    synthesis_prompt = generated_prompts[-1]
    assert "Where did the parent die?" in synthesis_prompt
    assert "Target City" in synthesis_prompt
    assert "UNTRUSTED SEARCH PLAN" not in synthesis_prompt
    assert output.extra_fields["response_text"] == "<answer>Target City</answer>"
    assert output.extra_fields["controller_synthesis_answer_source"] == "tagged"
    assert output.extra_fields["controller_synthesis_format_repaired"] is False
    assert output.extra_fields["controller_synthesis_format_valid"] is True
    assert output.extra_fields["controller_binding_valid_count"] == 1
    assert output.extra_fields["controller_binding_attempt_count"] == 1
    assert output.extra_fields["controller_binding_alias_count"] == 0
    assert output.extra_fields["controller_resolved_values_by_step"] == {
        1: {"step_1_result": "Parent Entity"}
    }
    assert dependent_turn.extra_fields["controller_resolved_values"] == {
        "step_1_result": "Parent Entity"
    }
    assert dependent_turn.extra_fields["controller_bound_query"] == (
        "Parent Entity target relation"
    )
    assert dependent_turn.extra_fields["controller_binding_valid"] is True
    turn_fields, _, _, train_fields = worker.gen_extra_fields([output], "opaque-id")
    assert turn_fields["controller_phase"] == ["hop", "synthesis"]
    assert train_fields["planner_turn_idx"] == [1, 2]


@pytest.mark.parametrize(
    "bad_dependent_output",
    [
        "<answer>premature intermediate result</answer>",
        "<search>{step_1_result} target relation</search>",
    ],
)
def test_controller_repairs_unusable_dependent_query(bad_dependent_output):
    worker = _controller_worker()
    prompt_ids = worker.tokenizer.encode(
        "<|im_start|>user\nQuestion: Where did the parent die?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    result = TeacherPlanResult(
        plan_id="plan-id",
        valid=True,
        plan=_multihop_plan(),
        raw_response="{}",
    )
    executed_queries = []

    async def parse_tool_call(text):
        match = re.search(r"<search>\s*(.*?)\s*</search>", text, re.DOTALL)
        requests = (
            [ToolRequest(name="search", arguments={"keyword": match.group(1)})]
            if match
            else []
        )
        return "", requests

    async def call_tool(request):
        executed_queries.append(request.arguments["keyword"])
        return ToolResponse(
            text=(
                "<information>\n[Doc 1](https://example.test/parent):\n"
                "Parent Entity\nDependency evidence.</information>"
            )
        )

    async def generate(prompt, sampling_params):
        del sampling_params
        prompt_text = worker.tokenizer.decode(prompt)
        text = (
            "The evidence names the city.</think><answer>Target City</answer>"
            if "CONTROLLED SYNTHESIS" in prompt_text
            else bad_dependent_output
        )
        return {"output_ids": worker.tokenizer.encode(text)}

    worker.toolcall_parser = parse_tool_call
    worker.tool_call = call_tool
    worker.generate = generate

    output = asyncio.run(
        worker.run_one_query(
            prompt_ids,
            answer="opaque-id",
            question_text="Where did the parent die?",
            guidance_mode="guided",
            teacher_plan_result=result,
        )
    )

    assert executed_queries[0] == "source entity relation"
    fallback_query = executed_queries[1]
    assert "Retrieve the requested fact" in fallback_query
    assert "source entity relation" in fallback_query
    assert "Parent Entity" in fallback_query
    assert "Where did the parent die?" not in fallback_query
    assert "step_1_result" not in fallback_query
    repaired_turn = output.single_turn_outputs[1]
    assert repaired_turn.extra_fields["controller_query_source"] == "fallback"
    assert repaired_turn.extra_fields["tool_call_repaired"] is True
    assert repaired_turn.extra_fields["is_terminal"] is False
    assert output.extra_fields["controller_fallback_query_count"] == 1
    assert output.extra_fields["controller_completed"] is True
    assert output.extra_fields["response_text"] == "<answer>Target City</answer>"
    assert "premature intermediate result" not in output.extra_fields["response_text"]


def test_controller_retries_invalid_binding_before_fallback():
    worker = _controller_worker()
    worker.controller_bind_max_attempts = 3
    prompt_ids = worker.tokenizer.encode(
        "<|im_start|>user\nQuestion: Where did the parent die?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    result = TeacherPlanResult(
        plan_id="plan-id",
        valid=True,
        plan=_multihop_plan(),
        raw_response="{}",
    )
    generated_prompts = []
    executed_queries = []

    async def parse_tool_call(text):
        match = re.search(r"<search>\s*(.*?)\s*</search>", text, re.DOTALL)
        return "", [ToolRequest(name="search", arguments={"keyword": match.group(1)})]

    async def call_tool(request):
        query = request.arguments["keyword"]
        executed_queries.append(query)
        return ToolResponse(
            text="<information>The source entity is Parent Entity.</information>"
        )

    async def generate(prompt, sampling_params):
        del sampling_params
        prompt_text = worker.tokenizer.decode(prompt)
        generated_prompts.append(prompt_text)
        if "CONTROLLED SYNTHESIS" in prompt_text:
            text = "Evidence supports it.</think><answer>Target City</answer>"
        elif "CONTROLLED BINDING RETRY" not in prompt_text:
            text = (
                '{"step_1_result":"Invented Entity"},'
                '"query":"Invented Entity target relation"}'
            )
        else:
            text = (
                '{"step_1_result":"Parent Entity"},'
                '"query":"Parent Entity target relation"}'
            )
        return {"output_ids": worker.tokenizer.encode(text)}

    worker.toolcall_parser = parse_tool_call
    worker.tool_call = call_tool
    worker.generate = generate

    output = asyncio.run(
        worker.run_one_query(
            prompt_ids,
            answer="opaque-id",
            question_text="Where did the parent die?",
            guidance_mode="guided",
            teacher_plan_result=result,
        )
    )

    assert executed_queries == [
        "source entity relation",
        "Parent Entity target relation",
    ]
    dependent_turn = output.single_turn_outputs[1]
    assert dependent_turn.extra_fields["controller_query_source"] == "policy"
    assert dependent_turn.extra_fields["controller_binding_attempts"] == 2
    assert dependent_turn.extra_fields["controller_binding_valid"] is True
    assert output.extra_fields["controller_binding_failure_reasons"] == {
        "ungrounded_value": 1
    }
    assert output.extra_fields["controller_fallback_query_count"] == 0
    assert "CONTROLLED BINDING RETRY" in generated_prompts[1]


def test_controller_fallback_uses_dependency_chain_and_evidence_titles():
    plan = _multihop_plan()
    query = controller_fallback_query(
        "Where did the parent die?",
        plan.steps[1],
        (plan.steps[0],),
        "[Doc 1](https://example.test/parent):\nParent Entity\nBody",
    )

    assert "Retrieve the requested fact" in query
    assert "Parent Entity" in query
    assert "source entity relation" in query
    assert "Where did the parent die?" not in query
    assert extract_controller_evidence_titles(
        "[Doc 1](https://example.test/parent):\nParent Entity\nBody"
    ) == ["Parent Entity"]


def test_controller_binding_requires_grounded_values_and_bound_query():
    step = _multihop_plan().steps[1]
    raw_response = json.dumps(
        {
            "resolved_values": {"step_1_result": "Parent Entity"},
            "query": "Parent Entity target relation",
        }
    )

    resolved, query, alias_used = extract_controller_bound_query(
        raw_response,
        step,
        "The source entity is Parent Entity.",
    )

    assert resolved == {"step_1_result": "Parent Entity"}
    assert query == "Parent Entity target relation"
    assert alias_used is False
    with pytest.raises(ValueError, match="not grounded"):
        extract_controller_bound_query(
            raw_response.replace("Parent Entity", "Invented Entity"),
            step,
            "The source entity is Parent Entity.",
        )
    with pytest.raises(ValueError, match="does not contain"):
        extract_controller_bound_query(
            raw_response.replace(
                "Parent Entity target relation", "generic target relation"
            ),
            step,
            "The source entity is Parent Entity.",
        )
    with pytest.raises(ValueError, match="trailing content"):
        extract_controller_bound_query(
            raw_response + "<answer>premature</answer>",
            step,
            "The source entity is Parent Entity.",
        )
    assert (
        classify_controller_binding_failure("<answer>guess</answer>", "bad")
        == "premature_answer"
    )
    assert (
        classify_controller_binding_failure("{step_1_result}", "bad")
        == "unresolved_placeholder"
    )
    assert (
        classify_controller_binding_failure('{"resolved_values":', "incomplete")
        == "empty_query"
    )


def test_controller_binding_does_not_cross_dependency_evidence():
    step = TeacherPlanStep(
        step_id=3,
        goal="Compare the two resolved entities",
        query_template="{step_1_result} {step_2_result} relation",
        expected_evidence="comparison evidence",
        depends_on=(1, 2),
    )
    crossed = json.dumps(
        {
            "resolved_values": {
                "step_1_result": "Beta Entity",
                "step_2_result": "Alpha Entity",
            },
            "query": "Beta Entity Alpha Entity relation",
        }
    )

    with pytest.raises(ValueError, match="not grounded"):
        extract_controller_bound_query(
            crossed,
            step,
            {1: "Alpha Entity evidence", 2: "Beta Entity evidence"},
        )


def test_controller_synthesis_normalizes_plain_and_rejects_search_output():
    normalized, source, valid = normalize_controller_synthesis_response(
        "<answer>Target City<|im_end|>"
    )
    invalid, invalid_source, invalid_valid = normalize_controller_synthesis_response(
        "<answer><search>another query</search>"
    )

    assert normalized == "<answer>Target City</answer>"
    assert source == "wrapped"
    assert valid is True
    assert invalid == "<answer></answer>"
    assert invalid_source == "empty"
    assert invalid_valid is False


def test_controller_separates_comparison_roots_and_enforces_answer_contract():
    worker = _controller_worker()
    prompt_ids = worker.tokenizer.encode(
        "<|im_start|>user\nQuestion: Which was earlier, Alpha or Beta?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    result = TeacherPlanResult(
        plan_id="comparison-plan",
        valid=True,
        plan=_comparison_plan(),
        raw_response="{}",
    )
    executed_queries = []
    synthesis_prompts = []

    async def parse_tool_call(text):
        match = re.search(r"<search>\s*(.*?)\s*</search>", text, re.DOTALL)
        return "", [ToolRequest(name="search", arguments={"keyword": match.group(1)})]

    async def call_tool(request):
        query = request.arguments["keyword"]
        executed_queries.append(query)
        return ToolResponse(text=f"<information>{query}: 1990</information>")

    async def generate(prompt, sampling_params):
        del sampling_params
        prompt_text = worker.tokenizer.decode(prompt)
        synthesis_prompts.append(prompt_text)
        return {"output_ids": worker.tokenizer.encode("<answer>Alpha</answer>")}

    worker.toolcall_parser = parse_tool_call
    worker.tool_call = call_tool
    worker.generate = generate

    output = asyncio.run(
        worker.run_one_query(
            prompt_ids,
            answer="opaque-id",
            question_text="Which was earlier, Alpha or Beta?",
            guidance_mode="guided",
            teacher_plan_result=result,
        )
    )

    assert executed_queries == ["Alpha release year", "Beta release year"]
    assert output.extra_fields["controller_template_query_count"] == 2
    assert len(synthesis_prompts) == 1
    assert "return the requested candidate/entity or yes/no" in synthesis_prompts[0]
    assert "Alpha release year" in synthesis_prompts[0]
    assert "Beta release year" in synthesis_prompts[0]
    assert output.extra_fields["response_text"] == "<answer>Alpha</answer>"
    assert output.extra_fields["controller_synthesis_answer_source"] == "tagged"


def test_missing_first_tool_call_forces_original_question_search():
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.cfg = OmegaConf.create({"agentloop": {"max_turns": 3}})
    worker.max_resp_len = 200
    worker.max_total_len = 2000
    worker.max_tool_response_length = 100
    worker.tool_response_truncate_side = "right"
    worker.print_outputs = False
    worker.tokenizer = _CharacterTokenizer()
    worker.force_search_on_first_turn = True
    worker.dual_query_retrieval = False
    executed_queries = []

    async def no_tool_call(_text):
        return "", []

    async def call_tool(request):
        executed_queries.append(request.arguments["keyword"])
        return ToolResponse(text="evidence")

    worker.toolcall_parser = no_tool_call
    worker.tool_call = call_tool
    llm_output = _turn(0, is_search=False, is_terminal=True)
    context = {
        "next_turn_id": 1,
        "last_llm_output": llm_output,
        "question_text": "original question",
        "teacher_rewrite_applied": False,
        "guidance_applied": False,
    }

    is_continue, _ = asyncio.run(
        worker.generate_tool_response(
            context,
            [],
            [1],
            [1],
            worker.tokenizer.encode("<answer>premature</answer>"),
            "<answer>premature</answer>",
        )
    )

    assert is_continue is True
    assert executed_queries == ["original question"]
    assert llm_output.extra_fields["search_query"] is None
    assert llm_output.extra_fields["tool_call_repaired"] is True
    assert llm_output.extra_fields["format_valid"] is False


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
            "decision": "KEEP",
            "plan_type": "singlehop",
            "steps": [],
        }
    )

    plan = parse_teacher_plan(response)

    assert plan.decision == "KEEP"
    assert plan.should_rewrite is False
    with pytest.raises(ValueError, match="start with one JSON object"):
        parse_teacher_plan(f"Here is the plan: {response}")
    with pytest.raises(ValueError, match="fields must be exactly"):
        parse_teacher_plan(response[:-1] + ', "answer": "secret"}')
    with pytest.raises(ValueError, match="KEEP plans require"):
        parse_teacher_plan(response.replace('"steps": []', '"steps": [{"step_id": 1}]'))


def test_teacher_multihop_gate_validates_dependencies_anchors_and_placeholders():
    question = "Where did Khanzada Begum's father die?"
    assert extract_query_anchors(question) == ("Khanzada Begum's",)
    safe_response = json.dumps(
        {
            "decision": "PLAN",
            "plan_type": "sequential",
            "steps": [
                {
                    "step_id": 1,
                    "goal": "Identify Khanzada Begum's father",
                    "query_template": "Khanzada Begum's father",
                    "expected_evidence": "the father's name",
                    "depends_on": [],
                },
                {
                    "step_id": 2,
                    "goal": "Find that individual's place of death",
                    "query_template": "{step_1_result} place of death",
                    "expected_evidence": "a stated place of death",
                    "depends_on": [1],
                },
            ],
        }
    )

    plan = parse_teacher_plan(safe_response, question=question)

    assert plan.should_rewrite is True
    assert plan.plan_type == "sequential"
    assert plan.steps[1].depends_on == (1,)
    unsafe_response = safe_response.replace(
        '"query_template": "Khanzada Begum\'s father"',
        '"query_template": "Khanzada Begum\'s father person"',
    )
    assert parse_teacher_plan(unsafe_response, question=question).should_plan is True

    dropped_anchor_response = safe_response.replace("Khanzada Begum's", "the subject")
    with pytest.raises(ValueError, match="drops protected anchors"):
        parse_teacher_plan(dropped_anchor_response, question=question)

    missing_placeholder_response = safe_response.replace(
        "{step_1_result} place of death", "intermediate entity place of death"
    )
    with pytest.raises(ValueError, match="placeholders must exactly match"):
        parse_teacher_plan(missing_placeholder_response, question=question)


def test_teacher_semantic_gate_requires_both_director_attribute_chains():
    question = "Do the directors of Alpha Film and Beta Film have the same nationality?"
    incomplete = TeacherPlan(
        decision="PLAN",
        plan_type="comparison",
        steps=(
            TeacherPlanStep(
                step_id=1,
                goal="Find Alpha Film's director",
                query_template="Alpha Film director",
                expected_evidence="director name",
                depends_on=(),
            ),
            TeacherPlanStep(
                step_id=2,
                goal="Find Beta Film's director",
                query_template="Beta Film director",
                expected_evidence="director name",
                depends_on=(),
            ),
            TeacherPlanStep(
                step_id=3,
                goal="Find the first director's nationality",
                query_template="{step_1_result} nationality",
                expected_evidence="nationality",
                depends_on=(1,),
            ),
        ),
    )
    complete = TeacherPlan(
        decision="PLAN",
        plan_type="comparison",
        steps=incomplete.steps
        + (
            TeacherPlanStep(
                step_id=4,
                goal="Find the second director's nationality",
                query_template="{step_2_result} nationality",
                expected_evidence="nationality",
                depends_on=(2,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="two dependent attribute hops"):
        validate_teacher_plan_semantics(question, incomplete)
    validate_teacher_plan_semantics(question, complete)


def test_label_only_plan_audit_requires_the_full_evidence_chain():
    plan = TeacherPlan(
        decision="PLAN",
        plan_type="sequential",
        steps=(
            TeacherPlanStep(
                step_id=1,
                goal="Identify Khanzada Begum's father",
                query_template="Khanzada Begum father",
                expected_evidence="father name",
                depends_on=(),
            ),
            TeacherPlanStep(
                step_id=2,
                goal="Find the father's place of death",
                query_template="{step_1_result} place of death",
                expected_evidence="place of death",
                depends_on=(1,),
            ),
        ),
    )
    evidences = [
        ["Khanzada Begum", "father", "Umar Shaikh Mirza II"],
        ["Umar Shaikh Mirza II", "place of death", "Fergana"],
    ]

    covered, edge_coverage, uncovered = audit_plan_semantic_coverage(plan, evidences)
    incomplete, incomplete_coverage, incomplete_edges = audit_plan_semantic_coverage(
        TeacherPlan(
            decision="PLAN",
            plan_type="comparison",
            steps=(
                plan.steps[0],
                TeacherPlanStep(
                    step_id=2,
                    goal="Find an unrelated occupation",
                    query_template="Khanzada Begum occupation",
                    expected_evidence="occupation",
                    depends_on=(),
                ),
            ),
        ),
        evidences,
    )

    assert covered is True
    assert edge_coverage == 1.0
    assert uncovered == []
    assert incomplete is False
    assert incomplete_coverage == 0.5
    assert incomplete_edges == [1]


def test_label_only_diagnostics_report_type_uplift_without_online_label_use():
    plan = TeacherPlan(
        decision="PLAN",
        plan_type="comparison",
        steps=(
            TeacherPlanStep(1, "Alpha release", "Alpha release year", "year", ()),
            TeacherPlanStep(2, "Beta release", "Beta release year", "year", ()),
        ),
    )
    results = []
    for mode, reward in (
        ("guided", 1.0),
        ("guided", 1.0),
        ("unguided", 0.0),
        ("unguided", 0.0),
    ):
        results.append(
            {
                "sample_id": 0,
                "guidance_mode": mode,
                "reward": reward,
                "answer_hit": bool(reward),
                "teacher_plan": plan.to_dict() if mode == "guided" else None,
                "turns": [
                    {
                        "visible_evidence": (
                            "Alpha was released in 1990. Beta was released in 1980."
                        )
                    }
                ],
            }
        )
    records = [
        {
            "query_id": "q0",
            "question_type": "comparison",
            "supporting_facts": [],
            "evidences": [
                ["Alpha", "publication date", "1990"],
                ["Beta", "publication date", "1980"],
            ],
        }
    ]

    metrics = build_label_only_diagnostics(
        results, records, bootstrap_seed=1234, bootstrap_samples=20
    )

    assert metrics["planner/plan_semantic_coverage_rate"] == 1.0
    assert metrics["planner/type/comparison/guided_EM"] == 1.0
    assert metrics["planner/type/comparison/guided_minus_unguided"] == 1.0
    assert metrics["search/type/comparison/guided_gold_evidence_object_coverage"] == 1.0
    assert all(result["question_type"] == "comparison" for result in results)


def test_abc_acceptance_metrics_require_every_subgate():
    metrics = {
        "planner/plan_valid_rate": 0.99,
        "planner/cache_hit_rate": 1.0,
        "planner/guided_controller_completion_rate": 0.99,
        "planner/guided_synthesis_format_valid_rate": 0.99,
        "search/guided_unresolved_placeholder_rate": 0.0,
        "planner/plan_semantic_coverage_rate": 0.95,
        "search/guided_dependent_query_binding_valid_rate": 0.95,
        "search/guided_controller_dependent_fallback_rate": 0.10,
        "planner/type/compositional_inference/guided_minus_unguided": 0.01,
        "planner/type/compositional_inference/uplift_ci_low": 0.0,
        "planner/guided_minus_unguided": 0.02,
        "planner/guided_uplift_ci_low": 0.001,
    }
    for question_type in (
        "compositional",
        "inference",
        "comparison",
        "bridge_comparison",
    ):
        metrics[f"planner/type/{question_type}/guided_minus_unguided"] = -0.01

    accepted = build_abc_acceptance_metrics(metrics)
    metrics["search/guided_dependent_query_binding_valid_rate"] = 0.949
    rejected = build_abc_acceptance_metrics(metrics)

    assert accepted["acceptance/ABC_pass"] == 1.0
    assert rejected["acceptance/B_dependent_binding"] == 0.0
    assert rejected["acceptance/ABC_pass"] == 0.0


def test_teacher_parser_normalizes_numeric_dependencies_and_trailing_artifacts():
    response = json.dumps(
        {
            "decision": "PLAN",
            "plan_type": "sequential",
            "steps": [
                {
                    "step_id": 1,
                    "goal": "Resolve the intermediate entity",
                    "query_template": "Khanzada Begum's father",
                    "expected_evidence": "the father's name",
                    "depends_on": [],
                },
                {
                    "step_id": 2,
                    "goal": "Retrieve the target fact",
                    "query_template": "{step_1_result} place of death",
                    "expected_evidence": "the place of death",
                    "depends_on": ["1"],
                },
            ],
        }
    )
    response = response[:-1] + " trailing generation artifact"

    plan = parse_teacher_plan(
        response, question="Where did Khanzada Begum's father die?"
    )

    assert plan.steps[1].depends_on == (1,)


def test_teacher_plan_cache_key_is_versioned_and_deterministic():
    first = teacher_plan_cache_key("question", "teacher-v1", 1234)

    assert first == teacher_plan_cache_key("question", "teacher-v1", 1234)
    assert first != teacher_plan_cache_key("question", "teacher-v2", 1234)
    assert first != teacher_plan_cache_key("question", "teacher-v1", 2025)


def test_legacy_four_field_cache_is_normalized_to_plan_steps():
    result = TeacherPlanResult.from_dict(
        {
            "plan_id": "legacy-id",
            "valid": True,
            "plan": {
                "decision": "REWRITE",
                "supplemental_query": "first direction",
                "expected_evidence": "supporting fact",
                "fallback_query": "second direction",
            },
            "raw_response": "{}",
        }
    )

    assert result.plan is not None
    assert result.plan.decision == "PLAN"
    assert result.plan.plan_type == "legacy"
    assert [step.query_template for step in result.plan.steps] == [
        "first direction",
        "second direction",
    ]


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
            "decision": "KEEP",
            "plan_type": "singlehop",
            "steps": [],
        }
    )
    generate_calls = []

    async def generate(prompt_ids, sampling_params, rollout_name):
        generate_calls.append((prompt_ids, sampling_params, rollout_name))
        assert "secret-ground-truth" not in tokenizer.decode(prompt_ids)
        return {"output_ids": tokenizer.encode(response)}

    first = asyncio.run(planner.get_plan("public question", generate))
    second = asyncio.run(planner.get_plan("public question", generate))
    reloaded = asyncio.run(
        FrozenTeacherPlanner(cfg, _TeacherTokenizer()).get_plan(
            "public question", generate
        )
    )

    assert first.valid is True
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert reloaded.cache_hit is True
    assert len(generate_calls) == 1
    assert tokenizer.last_messages[-1] == {
        "role": "user",
        "content": "public question",
    }


def test_frozen_teacher_retries_invalid_plan_with_question_only_prompt(tmp_path):
    tokenizer = _TeacherTokenizer()
    cfg = OmegaConf.create(
        {
            "data": {"seed": 1234},
            "teacher_planner": {
                "version": "teacher-v5",
                "seed": 1234,
                "cache_dir": str(tmp_path),
                "max_attempts": 3,
                "max_new_tokens": 64,
            },
        }
    )
    planner = FrozenTeacherPlanner(cfg, tokenizer)
    responses = iter(
        (
            "not JSON",
            json.dumps(
                {
                    "decision": "KEEP",
                    "plan_type": "singlehop",
                    "steps": [],
                }
            ),
        )
    )
    prompt_texts = []

    async def generate(prompt_ids, sampling_params, rollout_name):
        del sampling_params, rollout_name
        prompt_texts.append(tokenizer.decode(prompt_ids))
        return {"output_ids": tokenizer.encode(next(responses))}

    result = asyncio.run(planner.get_plan("public question", generate))

    assert result.valid is True
    assert len(prompt_texts) == 2
    assert "teacher plan must start with one JSON object" in prompt_texts[1]
    assert "Rejected plan: not JSON" in prompt_texts[1]
    assert "secret-ground-truth" not in prompt_texts[1]


def test_frozen_teacher_retries_keep_when_multihop_plan_is_required(tmp_path):
    tokenizer = _TeacherTokenizer()
    cfg = OmegaConf.create(
        {
            "data": {"seed": 1234},
            "teacher_planner": {
                "version": "teacher-v5",
                "seed": 1234,
                "cache_dir": str(tmp_path),
                "require_plan": True,
                "max_attempts": 2,
            },
        }
    )
    planner = FrozenTeacherPlanner(cfg, tokenizer)
    responses = iter(
        (
            json.dumps({"decision": "KEEP", "plan_type": "singlehop", "steps": []}),
            json.dumps(
                {
                    "decision": "PLAN",
                    "plan_type": "sequential",
                    "steps": [step.to_dict() for step in _multihop_plan().steps],
                }
            ),
        )
    )

    async def generate(*args, **kwargs):
        del args, kwargs
        return {"output_ids": tokenizer.encode(next(responses))}

    result = asyncio.run(planner.get_plan("public multi-hop question", generate))

    assert result.valid is True
    assert result.plan is not None
    assert result.plan.should_plan is True


def test_frozen_teacher_cache_only_rejects_misses(tmp_path):
    tokenizer = _TeacherTokenizer()
    cfg = OmegaConf.create(
        {
            "data": {"seed": 1234},
            "teacher_planner": {
                "version": "teacher-v1",
                "seed": 1234,
                "cache_dir": str(tmp_path),
                "cache_only": True,
                "max_new_tokens": 64,
            },
        }
    )
    planner = FrozenTeacherPlanner(cfg, tokenizer)

    async def unexpected_generate(*args, **kwargs):
        raise AssertionError((args, kwargs))

    with pytest.raises(RuntimeError, match="cache miss in cache-only mode"):
        asyncio.run(planner.get_plan("missing question", unexpected_generate))

    planner.cache_response(
        "cached question",
        json.dumps(
            {
                "decision": "KEEP",
                "plan_type": "singlehop",
                "steps": [],
            }
        ),
    )
    cached = asyncio.run(planner.get_plan("cached question", unexpected_generate))
    assert cached.valid is True
    assert cached.cache_hit is True


def test_teacher_plan_precompute_loads_jsonl_with_limit(tmp_path):
    data_path = tmp_path / "eval.jsonl"
    records = [
        {
            "prompt": (
                "<|im_start|>system\nExample Question: example only<|im_end|>\n"
                "<|im_start|>user\nQuestion: first question<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "answer": "private answer one",
        },
        {"prompt": "second question", "answer": "private answer two"},
    ]
    data_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )

    assert load_teacher_questions([str(data_path)], "prompt", data_size=1) == [
        "first question"
    ]


def test_extract_searchr1_question_accepts_raw_and_templated_prompts():
    assert extract_searchr1_question(" raw question ") == "raw question"
    assert (
        extract_searchr1_question(
            "Example Question: ignore this<|im_end|>\n"
            "<|im_start|>user\nQuestion: actual question?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        == "actual question?"
    )


def test_generic_guidance_matches_real_guidance_token_length():
    tokenizer = _CharacterTokenizer()
    plan = _multihop_plan()

    guided_ids = build_guidance_token_ids(tokenizer, plan, "guided")
    generic_ids = build_guidance_token_ids(tokenizer, plan, "generic")

    assert len(generic_ids) == len(guided_ids)
    assert generic_ids != guided_ids
    assert "<search" not in tokenizer.decode(guided_ids)


def test_teacher_guidance_requires_qwen_assistant_generation_prefix():
    tokenizer = _CharacterTokenizer()
    guidance_ids = tokenizer.encode("guidance")

    with pytest.raises(ValueError, match="must end with the Qwen ChatML assistant"):
        insert_guidance_user_message(
            tokenizer,
            tokenizer.encode("prompt without generation prefix"),
            guidance_ids,
        )


def test_keep_plan_leaves_policy_prompt_unguided():
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.max_prompt_len = 512
    worker.tokenizer = _CharacterTokenizer()
    worker.teacher_planner = type("Teacher", (), {"teacher_version": "teacher-v2"})()
    prompt_ids = worker.tokenizer.encode(
        "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n"
    )
    plan_result = TeacherPlanResult(
        plan_id="keep-plan",
        valid=True,
        plan=TeacherPlan("KEEP", "singlehop", ()),
        raw_response="{}",
    )

    processed_ids, context = asyncio.run(
        worker.pre_process_query(
            prompt_ids,
            "opaque-id",
            question_text="Question",
            guidance_mode="guided",
            teacher_plan_result=plan_result,
        )
    )

    assert processed_ids == prompt_ids
    assert context["teacher_decision"] == "KEEP"
    assert context["teacher_rewrite_applied"] is False
    assert context["guidance_applied"] is False


def test_shuffled_teacher_control_is_reproducible_derangement():
    plans = [
        TeacherPlanResult(
            plan_id=str(index),
            valid=True,
            plan=_multihop_plan(),
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


def test_multihop_teacher_guidance_persists_after_first_search():
    worker = Searchr1AgentLoopWorker.__new__(Searchr1AgentLoopWorker)
    worker.cfg = OmegaConf.create(
        {
            "agentloop": {"max_turns": 2},
            "rollout": {"model": {"model_path": "policy"}},
        }
    )
    worker.max_prompt_len = 512
    worker.max_resp_len = 64
    worker.max_total_len = 4096
    worker.max_tool_response_length = 8
    worker.tool_response_truncate_side = "right"
    worker.print_outputs = False
    worker.persist_teacher_plan = True
    worker.tokenizer = _CharacterTokenizer()
    worker.teacher_planner = type("Teacher", (), {"teacher_version": "teacher-v1"})()
    plan_result = TeacherPlanResult(
        plan_id="plan-id",
        valid=True,
        plan=_multihop_plan(),
        raw_response="{}",
    )
    original_prompt = worker.tokenizer.encode(
        "<|im_start|>system\nSystem prompt<|im_end|>\n"
        "<|im_start|>user\nQuestion<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
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
    guided_prompt_text = worker.tokenizer.decode(guided_prompt)

    assert guided_prompt_text.endswith("<|im_start|>assistant\n")
    assert guided_prompt_text.count("<|im_start|>assistant\n") == 1
    assert guided_prompt_text.index(
        "[BEGIN UNTRUSTED SEARCH PLAN]"
    ) < guided_prompt_text.rindex("<|im_start|>assistant\n")
    assert "<|im_start|>user\n[BEGIN UNTRUSTED SEARCH PLAN]" in guided_prompt_text

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
    assert next_prompt[: len(guided_prompt)] == guided_prompt
    assert "UNTRUSTED SEARCH PLAN" in worker.tokenizer.decode(next_prompt)


def test_shadow_metric_names_include_paired_uplift_and_controls():
    context = {
        "mode_reward_sums": defaultdict(
            float, {"guided": 3.0, "unguided": 1.0, "generic": 1.0}
        ),
        "mode_answer_hit_sums": defaultdict(
            float, {"guided": 4.0, "unguided": 2.0, "generic": 2.0}
        ),
        "mode_subem_sums": defaultdict(
            float, {"guided": 4.0, "unguided": 3.0, "generic": 3.0}
        ),
        "mode_tool_call_repair_sums": defaultdict(
            float, {"guided": 1.0, "unguided": 2.0, "generic": 0.0}
        ),
        "mode_dual_query_sums": defaultdict(
            float, {"guided": 2.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_controller_completion_sums": defaultdict(
            float, {"guided": 4.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_controller_applied_sums": defaultdict(
            float, {"guided": 4.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_controller_fallback_query_sums": defaultdict(
            float, {"guided": 2.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_controller_step_sums": defaultdict(
            float, {"guided": 8.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_controller_dependent_step_sums": defaultdict(
            float, {"guided": 2.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_controller_binding_valid_sums": defaultdict(
            float, {"guided": 2.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_controller_binding_attempt_sums": defaultdict(
            float, {"guided": 3.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_controller_binding_alias_sums": defaultdict(
            float, {"guided": 1.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_synthesis_format_repair_sums": defaultdict(
            float, {"guided": 3.0, "unguided": 0.0, "generic": 0.0}
        ),
        "mode_synthesis_format_valid_sums": defaultdict(
            float, {"guided": 4.0, "unguided": 0.0, "generic": 0.0}
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
        "plan_decision_by_id": {"a": "PLAN", "b": "KEEP"},
    }

    metrics = build_shadow_metrics(context, bootstrap_samples=200)

    assert metrics["eval/unguided_EM"] == 0.25
    assert metrics["planner/guided_EM"] == 0.75
    assert metrics["planner/guided_minus_unguided"] == 0.5
    assert metrics["planner/plan_valid_rate"] == 0.5
    assert metrics["planner/query_change_rate"] == 1.0
    assert metrics["planner/answer_hit_delta"] == 0.5
    assert metrics["planner/generic_minus_unguided"] == 0.0
    assert metrics["planner/rewrite_rate"] == 0.5
    assert metrics["planner/guided_diagnostic_SubEM"] == 1.0
    assert metrics["search/guided_dual_query_rate"] == 0.5
    assert metrics["search/guided_tool_call_repair_rate"] == 0.25
    assert metrics["planner/guided_controller_completion_rate"] == 1.0
    assert metrics["search/guided_controller_fallback_query_rate"] == 0.25
    assert metrics["search/guided_controller_dependent_fallback_rate"] == 1.0
    assert metrics["search/guided_dependent_query_binding_valid_rate"] == 1.0
    assert metrics["search/guided_binding_attempts_per_dependent_hop"] == 1.5
    assert metrics["search/guided_binding_alias_rate"] == 0.5
    assert metrics["planner/guided_synthesis_format_valid_rate"] == 1.0
    assert metrics["planner/guided_synthesis_format_repair_rate"] == 0.75
    assert metrics["planner/guided_uplift_ci_low"] == 0.5
    assert metrics["planner/guided_uplift_ci_high"] == 0.5
