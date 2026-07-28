# Copyright 2026 The RLinf Authors.
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
import json

from omegaconf import OmegaConf

from rlinf.agents.wideseek_r2.graph_memory.agent_loop import (
    WideSeekR2GraphAgentLoopWorker,
)
from rlinf.agents.wideseek_r2.graph_memory.schema import ActionState, GraphConfig
from rlinf.agents.wideseek_r2.graph_memory.state import (
    GraphRuntime,
    get_graph_runtime,
)
from rlinf.algorithms.toolcall_parsers import WideSeekR2GraphQwenToolCallParser
from rlinf.data.tool_call.tool_io_struct import ToolRequest


class _GraphDispatchWorker(WideSeekR2GraphAgentLoopWorker):
    def __init__(self):
        self.worker_calls = []

    async def worker_call(self, worker_request, main_task, sub_traj_id):
        self.worker_calls.append((worker_request, main_task, sub_traj_id))
        runtime = get_graph_runtime()
        action_id = worker_request.arguments["action_id"]
        runtime.mark_action_running(action_id, owner_sub_traj=sub_traj_id)
        runtime.mark_action_completed(action_id, summary="completed")
        return [], "completed", [sub_traj_id]


def _runtime() -> GraphRuntime:
    return GraphRuntime.bootstrap(
        question="q",
        answer_type="item",
        config={"enabled": True, "schema_version": "v2"},
    )


def _dispatch(worker, runtime, requests):
    runtime_token = runtime.context_token()
    try:
        return asyncio.run(
            worker._dispatch_planner_requests(
                requests,
                main_task="main",
                sub_traj_id=0,
                sub_traj_num=0,
            )
        )
    finally:
        runtime.reset_context(runtime_token)


def _subtask(prompt):
    return ToolRequest(
        name="subtask",
        arguments={"subtask": prompt, "focus_refs": ["entity:q"]},
    )


def test_invalid_tool_json_becomes_recoverable_parser_feedback():
    worker = object.__new__(WideSeekR2GraphAgentLoopWorker)
    worker.cfg = OmegaConf.create(
        {
            "agentloop": {
                "max_workers_per_planner": -1,
                "max_toolcall_per_worker": 5,
            }
        }
    )
    worker.toolcall_parser = WideSeekR2GraphQwenToolCallParser()
    response_text = (
        '<tool_call>{"name":"call_sub","arguments":'
        '{"subtasks":[{"subtask":"bad\\escape"}]}}</tool_call>'
    )

    requests, tool_call_info = asyncio.run(
        worker.extract_tool_calls(response_text, role="planner")
    )

    assert tool_call_info is None
    assert len(requests) == 1
    assert requests[0].name == "__graph_parser_error__"
    assert "Invalid \\escape" in requests[0].arguments["error"]

    runtime = _runtime()
    results = _dispatch(worker, runtime, requests)
    assert results == [([], None, [])]
    feedback = json.loads(runtime.graph_local_results[0])
    assert feedback["status"] == "GRAPH_PARSER_ERROR"
    assert "strict JSON" in feedback["instruction"]


def test_pure_graph_tool_call_does_not_enter_legacy_mas_metrics():
    worker = object.__new__(WideSeekR2GraphAgentLoopWorker)
    worker.cfg = OmegaConf.create(
        {
            "agentloop": {
                "max_workers_per_planner": -1,
                "max_toolcall_per_worker": 5,
            }
        }
    )
    worker.toolcall_parser = WideSeekR2GraphQwenToolCallParser()
    response_text = '<tool_call>{"name":"read_mem","arguments":{}}</tool_call>'

    requests, tool_call_info = asyncio.run(
        worker.extract_tool_calls(response_text, role="planner")
    )

    assert [request.name for request in requests] == ["read_mem"]
    assert tool_call_info is None


def test_dispatch_creates_flat_dynamic_action_and_runs_worker():
    runtime = _runtime()
    worker = _GraphDispatchWorker()

    results = _dispatch(worker, runtime, [_subtask("Find the answer")])

    assert results == [([], "completed", [1])]
    assert len(worker.worker_calls) == 1
    request, main_task, sub_traj_id = worker.worker_calls[0]
    assert request.arguments["action_id"] == "action:0:1"
    assert request.arguments["payload_graph_version"] == 0
    assert main_task == "main"
    assert sub_traj_id == 1
    assert runtime.activation_dag.actions["action:0:1"].state == ActionState.COMPLETED
    assert runtime.activation_dag.gates == {}
    assert runtime.activation_dag.joins == {}


def test_dispatch_returns_bounded_main_memory_result():
    runtime = _runtime()
    worker = _GraphDispatchWorker()

    results = _dispatch(
        worker,
        runtime,
        [ToolRequest(name="read_mem", arguments={"refs": [], "max_tokens": 32})],
    )

    assert results == [([], None, [])]
    payload = json.loads(runtime.graph_local_results[0])
    assert payload["status"] == "MEMORY_READ"
    assert payload["graph_version"] == 0
    assert payload["memory"] == []
    assert worker.worker_calls == []


def _finalize(worker, runtime, response_text):
    runtime_token = runtime.context_token()
    try:
        return asyncio.run(
            worker._finalize_trajectory(
                role="planner",
                response_text=response_text,
                turn_idx=0,
                total_turn_list=[],
                conv_id="test",
            )
        )
    finally:
        runtime.reset_context(runtime_token)


def test_phase1_finalization_preserves_main_response_without_audit_render():
    runtime = _runtime()
    worker = object.__new__(WideSeekR2GraphAgentLoopWorker)
    worker.graph_config = GraphConfig(enabled=True, schema_version="v2")
    worker.release_affinity = lambda _conv_id: None
    response_text = "<think>done</think>\nfinal answer<|im_end|>"

    answer_text, total_turn_list = _finalize(worker, runtime, response_text)

    assert answer_text == "<think>done</think>\nfinal answer"
    assert total_turn_list == [1]
    assert runtime.finish_requested is False
    assert runtime.last_audit is None
    assert runtime.answer_source == "main_response"


def test_phase1_after_role_loop_does_not_force_direct_answer_generation():
    worker = object.__new__(WideSeekR2GraphAgentLoopWorker)
    output_buffer = []

    response_text, turn_idx = asyncio.run(
        worker._after_role_loop(
            prompt_ids=[1],
            response_text="model response",
            role="planner",
            sub_traj_id=0,
            turn_idx=9,
            conv_id="test",
            output_buffer=output_buffer,
        )
    )

    assert response_text == "model response"
    assert turn_idx == 9
    assert output_buffer == []
