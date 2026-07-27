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

import pytest
from omegaconf import OmegaConf

from rlinf.agents.wideseek_r2.agent_loop import (
    get_wideseek_r2_agent_loop_cls,
)
from rlinf.agents.wideseek_r2.graph_memory.agent_loop import (
    WideSeekR2GraphAgentLoopWorker,
)
from rlinf.agents.wideseek_r2.wideseek_r2 import WideSeekR2AgentLoopWorker
from rlinf.data.tool_call.tool_io_struct import ToolRequest, ToolResponse


class _RecordingTokenizer:
    def __init__(self):
        self.chat_template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_calls.append((messages, kwargs))
        return [1, 2, 3, 4]

    def decode(self, _token_ids):
        return "answer"


class _RecordingDispatchWorker(WideSeekR2AgentLoopWorker):
    def __init__(self):
        self.worker_calls = []
        self.tool_calls = []
        self.use_access_summary = False

    async def worker_call(self, worker_request, main_task, sub_traj_id):
        self.worker_calls.append((worker_request, main_task, sub_traj_id))
        return [], f"summary-{sub_traj_id}", [sub_traj_id]

    async def tool_call(self, tool_request):
        self.tool_calls.append(tool_request)
        return ToolResponse(text=f"result-{tool_request.name}")


class _SingleTurnWorker(WideSeekR2AgentLoopWorker):
    def __init__(self):
        self.cfg = OmegaConf.create(
            {"agentloop": {"max_sa_turns": 1, "add_few_shot": False}}
        )
        self.max_total_len = 32
        self.fixed_role = None
        self.use_fixed_rollout = False
        self.return_logprobs = True
        self.tokenizer = _RecordingTokenizer()
        self.released_conversations = []

    async def generate(self, prompt_ids, **kwargs):
        assert prompt_ids == [1, 2, 3, 4]
        return {
            "output_ids": [5],
            "finish_reason": "stop",
            "logprobs": [0.5],
        }

    async def extract_tool_calls(self, response_text, role):
        assert response_text == "answer"
        assert role == "single"
        return [], None

    def release_affinity(self, conv_id):
        self.released_conversations.append(conv_id)


class _TraceTokenizer:
    """Tokenizer stub with stable decodes for the characterization trace."""

    _decoded_responses = {
        (10,): "planner_tool",
        (11,): "planner_final",
        (20,): "worker_tool",
        (21,): "<answer>worker summary</answer>",
        (30,): "single_tool",
        (31,): "single_final",
    }

    def decode(self, token_ids):
        token_ids = tuple(token_ids)
        return self._decoded_responses.get(
            token_ids,
            "prompt:" + ",".join(str(token_id) for token_id in token_ids),
        )


class _TraceWorker(WideSeekR2AgentLoopWorker):
    """Deterministic worker used to lock the pre-hook execution contract."""

    _initial_prompt_ids = {"planner": [100], "worker": [200], "single": [300]}

    def __init__(self):
        self.cfg = OmegaConf.create(
            {
                "agentloop": {
                    "max_planner_turns": 2,
                    "max_worker_turns": 2,
                    "max_sa_turns": 2,
                    "max_workers_per_planner": -1,
                    "max_toolcall_per_worker": 5,
                    "add_few_shot": False,
                }
            }
        )
        self.max_total_len = 64
        self.fixed_role = None
        self.use_fixed_rollout = False
        self.return_logprobs = True
        self.use_access_summary = False
        self.tokenizer = _TraceTokenizer()
        self.turns_by_session = {}
        self.contexts = []
        self.generations = []
        self.worker_dispatches = []
        self.tool_calls = []
        self.feedback = []
        self.released_count = 0

    def _build_role_context(
        self,
        *,
        origin_question,
        role,
        add_few_shot,
        max_workers_per_planner,
        max_toolcall_per_worker,
        main_task,
        answer_type,
        max_turns,
    ):
        del (
            add_few_shot,
            max_workers_per_planner,
            max_toolcall_per_worker,
            main_task,
        )
        self.contexts.append((role, origin_question, answer_type, max_turns))
        return list(self._initial_prompt_ids[role])

    async def generate(self, prompt_ids, *, session_id, **kwargs):
        del kwargs
        role = next(
            role
            for role, initial_prompt_ids in self._initial_prompt_ids.items()
            if prompt_ids[0] == initial_prompt_ids[0]
        )
        turn_idx = self.turns_by_session.get(session_id, 0)
        self.turns_by_session[session_id] = turn_idx + 1

        response_ids_by_role = {
            "planner": [[10], [11]],
            "worker": [[20], [21]],
            "single": [[30], [31]],
        }
        response_ids = response_ids_by_role[role][turn_idx]
        self.generations.append(
            (role, turn_idx, tuple(prompt_ids), tuple(response_ids))
        )
        return {
            "output_ids": response_ids,
            "finish_reason": "stop",
            "logprobs": [0.5],
        }

    async def extract_tool_calls(self, response_text, role):
        if response_text == "planner_tool":
            requests = [
                ToolRequest(
                    name="subtask",
                    arguments={"subtask": "subtask-a"},
                ),
                ToolRequest(
                    name="subtask",
                    arguments={"subtask": "subtask-b"},
                ),
            ]
        elif response_text in {"worker_tool", "single_tool"}:
            requests = [
                ToolRequest(
                    name="search",
                    arguments={"query": "shared-query"},
                ),
                ToolRequest(
                    name="access",
                    arguments={
                        "url": "https://example.test",
                        "info_to_extract": "target-fact",
                    },
                ),
            ]
        else:
            requests = []

        if not requests:
            return [], None

        tool_call_info = {
            "subtask": sum(request.name == "subtask" for request in requests),
            "search": sum(request.name == "search" for request in requests),
            "access": sum(request.name == "access" for request in requests),
            "role": role,
        }
        return requests, tool_call_info

    async def worker_call(self, worker_request, main_task, sub_traj_id):
        self.worker_dispatches.append(
            (worker_request.arguments["subtask"], main_task, sub_traj_id)
        )
        return await super().worker_call(worker_request, main_task, sub_traj_id)

    async def tool_call(self, tool_request):
        self.tool_calls.append(
            (tool_request.name, tuple(sorted(tool_request.arguments.items())))
        )
        return ToolResponse(text=f"{tool_request.name}-result")

    def get_tool_response_ids(self, tool_messages):
        self.feedback.append(tool_messages[0]["content"])
        return [900]

    def release_affinity(self, conv_id):
        del conv_id
        self.released_count += 1


_EXPECTED_WORKER_FEEDBACK = (
    "# Search query:\n"
    "shared-query\n"
    "# Result:\n"
    "search-result\n\n"
    "# Access URL:\n"
    "https://example.test\n"
    "# Result:\n"
    "access-result\n\n"
    "Your next answer will be on turn 2. "
    "You MUST finish the entire answer by turn 2."
)


_EXPECTED_PLANNER_FEEDBACK = (
    "# Subtask 1:\n"
    "subtask-a\n"
    "# Result:\n"
    "worker summary\n\n"
    "# Subtask 2:\n"
    "subtask-b\n"
    "# Result:\n"
    "worker summary\n\n"
    "Your next answer will be on turn 2. "
    "You MUST finish the entire answer by turn 2."
)


def _trace_snapshot(worker, result):
    output_buffer, answer_text, total_turn_list = result
    return {
        "contexts": sorted(worker.contexts),
        "generations": sorted(worker.generations),
        "worker_dispatches": worker.worker_dispatches,
        "tool_calls": sorted(worker.tool_calls),
        "feedback": sorted(worker.feedback),
        "outputs": [
            {
                "prompt_ids": output.prompt_ids,
                "response_ids": output.response_ids,
                "response_text": output.response_text,
                "role": output.extra_fields["role"],
                "sub_traj_id": output.extra_fields["idx_to_sub_traj"],
                "tool_call_info": output.tool_call_info,
            }
            for output in output_buffer
        ],
        "answer_text": answer_text,
        "total_turn_list": total_turn_list,
        "released_count": worker.released_count,
    }


def test_mas_hook_refactor_preserves_fixed_model_trace():
    """The MAS hook refactor preserves the pre-refactor golden execution trace."""
    worker = _TraceWorker()
    result = asyncio.run(
        worker.run_one_query_role(
            question="main-question",
            role="planner",
            sub_traj_id=0,
            answer_type="table",
        )
    )

    assert _trace_snapshot(worker, result) == {
        "contexts": [
            ("planner", "main-question", "table", 2),
            ("worker", "subtask-a", None, 2),
            ("worker", "subtask-b", None, 2),
        ],
        "generations": [
            ("planner", 0, (100,), (10,)),
            ("planner", 1, (100, 10, 900), (11,)),
            ("worker", 0, (200,), (20,)),
            ("worker", 0, (200,), (20,)),
            ("worker", 1, (200, 20, 900), (21,)),
            ("worker", 1, (200, 20, 900), (21,)),
        ],
        "worker_dispatches": [
            ("subtask-a", "main-question", 1),
            ("subtask-b", "main-question", 2),
        ],
        "tool_calls": [
            (
                "access",
                (("info_to_extract", "target-fact"), ("url", "https://example.test")),
            ),
            (
                "access",
                (("info_to_extract", "target-fact"), ("url", "https://example.test")),
            ),
            ("search", (("query", "shared-query"),)),
            ("search", (("query", "shared-query"),)),
        ],
        "feedback": [
            _EXPECTED_WORKER_FEEDBACK,
            _EXPECTED_WORKER_FEEDBACK,
            _EXPECTED_PLANNER_FEEDBACK,
        ],
        "outputs": [
            {
                "prompt_ids": [100],
                "response_ids": [10],
                "response_text": "planner_tool",
                "role": "planner",
                "sub_traj_id": 0,
                "tool_call_info": {
                    "subtask": 2,
                    "search": 0,
                    "access": 0,
                    "role": "planner",
                },
            },
            {
                "prompt_ids": [200],
                "response_ids": [20],
                "response_text": "worker_tool",
                "role": "worker",
                "sub_traj_id": 1,
                "tool_call_info": {
                    "subtask": 0,
                    "search": 1,
                    "access": 1,
                    "role": "worker",
                },
            },
            {
                "prompt_ids": [200, 20, 900],
                "response_ids": [21],
                "response_text": "<answer>worker summary</answer>",
                "role": "worker",
                "sub_traj_id": 1,
                "tool_call_info": None,
            },
            {
                "prompt_ids": [200],
                "response_ids": [20],
                "response_text": "worker_tool",
                "role": "worker",
                "sub_traj_id": 2,
                "tool_call_info": {
                    "subtask": 0,
                    "search": 1,
                    "access": 1,
                    "role": "worker",
                },
            },
            {
                "prompt_ids": [200, 20, 900],
                "response_ids": [21],
                "response_text": "<answer>worker summary</answer>",
                "role": "worker",
                "sub_traj_id": 2,
                "tool_call_info": None,
            },
            {
                "prompt_ids": [100, 10, 900],
                "response_ids": [11],
                "response_text": "planner_final",
                "role": "planner",
                "sub_traj_id": 0,
                "tool_call_info": None,
            },
        ],
        "answer_text": "planner_final",
        "total_turn_list": [2, 2, 2],
        "released_count": 3,
    }


def test_single_agent_hook_refactor_preserves_fixed_model_trace():
    """The SA hook refactor preserves the pre-refactor tool and output trace."""
    worker = _TraceWorker()
    result = asyncio.run(
        worker.run_one_query_role(
            question="single-question",
            role="single",
            sub_traj_id=0,
            answer_type="table",
        )
    )

    snapshot = _trace_snapshot(worker, result)
    assert snapshot["contexts"] == [("single", "single-question", "table", 2)]
    assert snapshot["generations"] == [
        ("single", 0, (300,), (30,)),
        ("single", 1, (300, 30, 900), (31,)),
    ]
    assert snapshot["worker_dispatches"] == []
    assert snapshot["tool_calls"] == [
        (
            "access",
            (("info_to_extract", "target-fact"), ("url", "https://example.test")),
        ),
        ("search", (("query", "shared-query"),)),
    ]
    assert snapshot["feedback"] == [_EXPECTED_WORKER_FEEDBACK]
    assert snapshot["outputs"] == [
        {
            "prompt_ids": [300],
            "response_ids": [30],
            "response_text": "single_tool",
            "role": "single",
            "sub_traj_id": 0,
            "tool_call_info": {
                "subtask": 0,
                "search": 1,
                "access": 1,
                "role": "single",
            },
        },
        {
            "prompt_ids": [300, 30, 900],
            "response_ids": [31],
            "response_text": "single_final",
            "role": "single",
            "sub_traj_id": 0,
            "tool_call_info": None,
        },
    ]
    assert snapshot["answer_text"] == "single_final"
    assert snapshot["total_turn_list"] == [2]
    assert snapshot["released_count"] == 1


@pytest.mark.parametrize("workflow", ["mas", "sa", "legacy"])
def test_workflow_factory_preserves_existing_worker(workflow):
    assert get_wideseek_r2_agent_loop_cls(workflow) is WideSeekR2AgentLoopWorker


def test_workflow_factory_exposes_graph_extension_point():
    assert get_wideseek_r2_agent_loop_cls("mas_graph") is WideSeekR2GraphAgentLoopWorker


def test_build_role_context_keeps_prompt_builder_contract(monkeypatch):
    worker = object.__new__(WideSeekR2AgentLoopWorker)
    worker.max_total_len = 3
    worker.tokenizer = _RecordingTokenizer()
    captured = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return ([{"role": "user", "content": "question"}], [{"name": "tool"}])

    monkeypatch.setattr(
        "rlinf.agents.wideseek_r2.wideseek_r2.build_message_history_and_tools",
        fake_builder,
    )

    prompt_ids = worker._build_role_context(
        origin_question="question",
        role="planner",
        add_few_shot=False,
        max_workers_per_planner=4,
        max_toolcall_per_worker=5,
        main_task=None,
        answer_type="table",
        max_turns=7,
    )

    assert prompt_ids == [1, 2, 3]
    assert captured == {
        "origin_question": "question",
        "role": "planner",
        "add_few_shot": False,
        "max_workers_per_planner": 4,
        "max_toolcall_per_worker": 5,
        "main_task": None,
        "answer_type": "table",
    }
    messages, kwargs = worker.tokenizer.chat_template_calls[0]
    assert messages[-1]["content"].endswith("within 7 turns")
    assert kwargs["tools"] == [{"name": "tool"}]


def test_planner_dispatch_keeps_order_and_subtrajectory_ids():
    worker = _RecordingDispatchWorker()
    requests = [
        ToolRequest(name="subtask", arguments={"subtask": "first"}),
        ToolRequest(name="subtask", arguments={"subtask": "second"}),
    ]

    results = asyncio.run(
        worker._dispatch_planner_requests(
            requests,
            main_task="main",
            sub_traj_id=0,
            sub_traj_num=3,
        )
    )

    assert [call[1:] for call in worker.worker_calls] == [
        ("main", 4),
        ("main", 5),
    ]
    assert [result[1] for result in results] == ["summary-4", "summary-5"]


def test_planner_feedback_keeps_success_failure_messages_and_turn_hint():
    worker = object.__new__(WideSeekR2AgentLoopWorker)
    requests = [
        ToolRequest(name="subtask", arguments={"subtask": "first"}),
        ToolRequest(name="subtask", arguments={"subtask": "second"}),
    ]
    results = [([], "summary", [2]), ([], None, [3])]

    worker_buffer, worker_turns, messages = worker._format_planner_feedback(
        requests,
        results,
        turn_idx=0,
        max_turns=5,
    )

    assert worker_buffer == []
    assert worker_turns == [2, 3]
    assert messages[0]["role"] == "tool"
    assert "# Subtask 1:\nfirst\n# Result:\nsummary" in messages[0]["content"]
    assert "# Subtask 2:\nsecond\n# Result:" in messages[0]["content"]
    assert "Your next answer will be on turn 2." in messages[0]["content"]


def test_worker_dispatch_and_feedback_keep_tool_order():
    worker = _RecordingDispatchWorker()
    requests = [
        ToolRequest(name="search", arguments={"query": "one"}),
        ToolRequest(
            name="access",
            arguments={"url": "https://example.test", "info_to_extract": "fact"},
        ),
    ]

    responses = asyncio.run(worker._dispatch_worker_requests(requests))
    messages = asyncio.run(
        worker._format_worker_feedback(
            requests,
            responses,
            turn_idx=1,
            max_turns=5,
        )
    )

    assert worker.tool_calls == requests
    assert [response.text for response in responses] == [
        "result-search",
        "result-access",
    ]
    assert messages[0]["role"] == "tool"
    assert "# Search query:\none\n# Result:\nresult-search" in messages[0]["content"]
    assert (
        "# Access URL:\nhttps://example.test\n# Result:\nresult-access"
        in messages[0]["content"]
    )
    assert "Your next answer will be on turn 3." in messages[0]["content"]


def test_single_turn_default_hooks_preserve_finalization():
    worker = _SingleTurnWorker()

    output_buffer, answer_text, total_turns = asyncio.run(
        worker.run_one_query_role(
            question="question",
            role="single",
            sub_traj_id=0,
            answer_type="table",
        )
    )

    assert len(output_buffer) == 1
    assert output_buffer[0].prompt_ids == [1, 2, 3, 4]
    assert output_buffer[0].response_ids == [5]
    assert answer_text == "answer"
    assert total_turns == [1]
    assert len(worker.released_conversations) == 1
