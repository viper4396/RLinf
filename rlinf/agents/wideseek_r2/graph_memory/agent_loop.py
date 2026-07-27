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

"""WideSeek-R2 ``mas_graph`` agent loop for the Phase 1 item MVP."""

from __future__ import annotations

import asyncio
from typing import Optional

from omegaconf import DictConfig

from rlinf.agents.wideseek_r2.graph_memory.credit import graph_credit_metrics
from rlinf.agents.wideseek_r2.graph_memory.prompts import (
    GRAPH_PLANNER_GUIDANCE,
    GRAPH_WORKER_GUIDANCE,
)
from rlinf.agents.wideseek_r2.graph_memory.renderer import audit_item, render
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionState,
    AuditResult,
    GapReport,
    GraphConfig,
)
from rlinf.agents.wideseek_r2.graph_memory.state import (
    GraphRuntime,
    GraphStateError,
    get_current_action,
    get_current_phase,
    get_graph_runtime,
)
from rlinf.agents.wideseek_r2.graph_memory.tools import (
    GraphToolExecutor,
    format_activation_event,
    get_graph_tools_description,
)
from rlinf.agents.wideseek_r2.wideseek_r2 import WideSeekR2AgentLoopWorker
from rlinf.data.tool_call.tool_io_struct import ToolRequest, ToolResponse


class WideSeekR2GraphAgentLoopWorker(WideSeekR2AgentLoopWorker):
    """Graph-memory worker with trajectory-local state and scoped activation."""

    def __init__(self, cfg: DictConfig, placement):
        graph_cfg = cfg.agentloop.get("graph_memory", None)
        graph_config = GraphConfig.from_config(graph_cfg)
        if not graph_config.enabled:
            raise ValueError(
                "workflow=mas_graph requires agentloop.graph_memory.enabled=true"
            )
        if graph_config.schema_version != "v1":
            raise ValueError(
                "workflow=mas_graph Phase 1 requires graph_memory.schema_version=v1"
            )
        if graph_config.condition_dsl_version != "v1":
            raise ValueError(
                "workflow=mas_graph Phase 1 requires condition_dsl_version=v1"
            )
        parser_name = str(cfg.agentloop.get("toolcall_parser", ""))
        if parser_name != "wideseek_r2-graph-qwen":
            raise ValueError(
                "workflow=mas_graph requires toolcall_parser=wideseek_r2-graph-qwen"
            )
        if not graph_config.deterministic_render:
            raise ValueError("mas_graph requires deterministic_render=true")
        super().__init__(cfg, placement)
        self.graph_config = graph_config

        # Keep existing per-turn/traj metadata and add bounded graph fields.
        self.extra_keys_turn = list(self.extra_keys_turn) + [
            "graph_version_before",
            "graph_version_after",
            "action_id",
            "activation_event_ids",
            "evidence_proposal_id",
            "accepted_node_count",
            "rejected_node_count",
            "read_evidence_token_count",
            "graph_tool_phase",
        ]
        self.extra_keys_traj = list(self.extra_keys_traj) + [
            "task_contract",
            "graph_summary",
            "activation_summary",
            "completion_audit",
            "render_fact_refs",
            "graph_metrics",
        ]

    @property
    def runtime(self) -> GraphRuntime:
        return get_graph_runtime()

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
        """Build the existing prompt plus static graph role constraints."""

        from rlinf.agents.wideseek_r2.utils.prompt_utils import (
            build_message_history_and_tools,
            get_first_turn_hint,
        )

        message_history, tools = build_message_history_and_tools(
            origin_question=origin_question,
            role=role,
            add_few_shot=add_few_shot,
            max_workers_per_planner=max_workers_per_planner,
            max_toolcall_per_worker=max_toolcall_per_worker,
            main_task=main_task,
            answer_type=answer_type,
        )
        guidance = (
            GRAPH_PLANNER_GUIDANCE if role == "planner" else GRAPH_WORKER_GUIDANCE
        )
        message_history[0]["content"] += "\n\n" + guidance
        graph_tools = get_graph_tools_description(
            role,
            max_workers_per_planner=max_workers_per_planner,
            max_toolcall_per_worker=max_toolcall_per_worker,
        )
        # Replace the legacy ``create_sub_agents`` schema with the structured
        # action-scoped version; simply appending would leave the model seeing
        # two definitions with the same function name.
        graph_by_name = {
            tool["function"]["name"]: tool
            for tool in graph_tools
            if "function" in tool and "name" in tool["function"]
        }
        replaced_names = set()
        for index, tool in enumerate(tools):
            name = tool.get("function", {}).get("name")
            if name in graph_by_name:
                tools[index] = graph_by_name[name]
                replaced_names.add(name)
        tools.extend(
            tool for name, tool in graph_by_name.items() if name not in replaced_names
        )
        turn_hint = get_first_turn_hint(max_turns=max_turns)
        assert message_history[-1]["role"] == "user"
        message_history[-1]["content"] += turn_hint
        prompt_ids = self.tokenizer.apply_chat_template(
            message_history,
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
        )
        return prompt_ids[: self.max_total_len]

    async def extract_tool_calls(self, response_text: str, role: str):
        """Parse graph tools and surface parser errors instead of dropping them."""

        result = await super().extract_tool_calls(response_text, role)
        parser_error = getattr(self.toolcall_parser, "last_error", None)
        if parser_error:
            raise ValueError(f"WideSeek-R2 graph tool parser error: {parser_error}")
        return result

    async def _before_role_turn(
        self,
        *,
        prompt_ids: list[int],
        role: str,
        turn_idx: int,
        conv_id: str,
    ) -> list[int]:
        """Inject pending activation deltas only at the generation boundary."""

        del turn_idx, conv_id
        runtime = self.runtime
        runtime.begin_turn()
        action_id = get_current_action() if role == "worker" else None
        events = []
        if action_id:
            events.extend(runtime.pending_events(action_id, consume=True))
        else:
            for candidate_action_id in sorted(runtime.event_queues):
                planner_events = runtime.pending_events(
                    candidate_action_id, consume=False
                )
                events.extend(
                    event
                    for event in planner_events
                    if event.event_id not in runtime.planner_seen_event_ids
                )
                runtime.planner_seen_event_ids.update(
                    event.event_id for event in planner_events
                )
        if not events:
            return prompt_ids
        event_text = "\n\n".join(format_activation_event(event) for event in events)
        event_ids = self.get_tool_response_ids(
            [{"role": "tool", "content": event_text}]
        )
        return (prompt_ids + event_ids)[: self.max_total_len]

    async def _dispatch_planner_requests(
        self,
        tool_requests: list[ToolRequest],
        *,
        main_task: str,
        sub_traj_id: int,
        sub_traj_num: int,
    ) -> list[tuple[list, Optional[str], list]]:
        """Execute planner graph tools and concurrently dispatch ready actions."""

        if sub_traj_id != 0:
            raise GraphStateError("Planner must own sub-trajectory 0")
        self.runtime.graph_local_results = {}
        worker_indexes = []
        worker_tasks = []
        results: list[tuple[list, Optional[str], list]] = [([], None, [])] * len(
            tool_requests
        )
        for index, request in enumerate(tool_requests):
            if request.name == "subtask":
                action_id = str(request.arguments.get("action_id", ""))
                if not action_id:
                    raise ValueError("Graph subtask requires action_id")
                action = self.runtime.activation_dag.actions.get(action_id)
                if action_id in {"action:plan_task", "action:initial_frontier"}:
                    raise ValueError(
                        f"Graph action {action_id!r} is system-owned and cannot be delegated"
                    )
                if action is None or action.state != ActionState.READY:
                    raise ValueError(
                        f"Graph action {action_id!r} is not in the ready frontier"
                    )
                worker_indexes.append(index)
                worker_tasks.append(
                    self.worker_call(
                        request,
                        main_task,
                        sub_traj_id + len(worker_tasks) + 1 + sub_traj_num,
                    )
                )
            else:
                self.runtime.set_phase("graph")
                response = await self._execute_graph_tool(
                    request,
                    role="planner",
                )
                self.runtime.graph_local_results[index] = response.text
        if worker_tasks:
            worker_results = await asyncio.gather(*worker_tasks)
            for index, result in zip(worker_indexes, worker_results):
                results[index] = result
        return results

    def _format_planner_feedback(
        self,
        tool_requests: list[ToolRequest],
        worker_results: list[tuple[list, Optional[str], list]],
        *,
        turn_idx: int,
        max_turns: int,
    ):
        """Return worker summaries plus bounded local graph-tool responses."""

        from rlinf.agents.wideseek_r2.utils.prompt_utils import (
            get_next_turn_hint,
            get_planner_subtask_failed_message,
            get_planner_subtask_result_message,
        )

        worker_buffer = []
        worker_turn_list = []
        messages = []
        worker_number = 0
        for index, request in enumerate(tool_requests):
            if request.name != "subtask":
                messages.append(
                    "# Graph tool result:\n"
                    + self.runtime.graph_local_results.get(index, "")
                )
                continue
            worker_number += 1
            worker_outputs, summary, turns = worker_results[index]
            worker_buffer.extend(worker_outputs)
            worker_turn_list.extend(turns)
            subtask = request.arguments.get("subtask", "")
            if summary is not None:
                messages.append(
                    get_planner_subtask_result_message(worker_number, subtask, summary)
                )
            else:
                messages.append(
                    get_planner_subtask_failed_message(worker_number, subtask)
                )
        messages.append(
            get_next_turn_hint(next_turn_idx=turn_idx + 2, max_turns=max_turns)
        )
        return (
            worker_buffer,
            worker_turn_list,
            [{"role": "tool", "content": "\n\n".join(messages)}],
        )

    async def _dispatch_worker_requests(
        self, tool_requests: list[ToolRequest]
    ) -> list[ToolResponse]:
        """Execute one graph-local or external phase, never a mixed phase."""

        graph_names = {
            "read_evidence",
            "submit_evidence",
            "report_action_status",
            "propose_next_actions",
        }
        phases = {
            "graph" if request.name in graph_names else "external"
            for request in tool_requests
        }
        if len(phases) > 1 and self.graph_config.reject_mixed_tool_phases:
            raise ValueError("Mixed graph/external worker tool phases are not allowed")
        if phases == {"graph"}:
            self.runtime.set_phase("graph")
            return await asyncio.gather(
                *(
                    self._execute_graph_tool(request, role="worker")
                    for request in tool_requests
                )
            )
        self.runtime.set_phase("external")
        return await super()._dispatch_worker_requests(tool_requests)

    async def _format_worker_feedback(
        self,
        tool_requests: list[ToolRequest],
        tool_responses: list[ToolResponse],
        *,
        turn_idx: int,
        max_turns: int,
    ) -> list[dict]:
        """Format graph-local responses without exposing the full graph."""

        if all(
            request.name
            in {
                "read_evidence",
                "submit_evidence",
                "report_action_status",
                "propose_next_actions",
            }
            for request in tool_requests
        ):
            from rlinf.agents.wideseek_r2.utils.prompt_utils import get_next_turn_hint

            text = "\n\n".join(
                f"# Graph tool {request.name}:\n{response.text}"
                for request, response in zip(tool_requests, tool_responses)
            )
            text += get_next_turn_hint(turn_idx + 2, max_turns)
            return [{"role": "tool", "content": text}]
        return await super()._format_worker_feedback(
            tool_requests,
            tool_responses,
            turn_idx=turn_idx,
            max_turns=max_turns,
        )

    async def _execute_graph_tool(
        self, request: ToolRequest, *, role: str
    ) -> ToolResponse:
        # Resolve the executor from the ContextVar-local runtime on every call;
        # a Ray actor may serve more than one rollout concurrently.
        executor = GraphToolExecutor(self.runtime)
        return await executor.execute(
            request,
            role=role,
            action_id=get_current_action(),
        )

    async def run_one_query_role(
        self,
        question: str,
        role: str,
        sub_traj_id: int,
        main_task: str | None = None,
        answer_type: str | None = None,
    ):
        """Run the shared role loop and attach bounded graph turn metadata."""

        graph_version_before = self.runtime.version
        action_id = get_current_action()
        output_buffer, answer_text, total_turn_list = await super().run_one_query_role(
            question=question,
            role=role,
            sub_traj_id=sub_traj_id,
            main_task=main_task,
            answer_type=answer_type,
        )
        event_ids = sorted(
            self.runtime.delivered_event_keys
            if role == "worker"
            else self.runtime.planner_seen_event_ids
        )
        event_limit = self.graph_config.max_delta_events_per_turn
        if event_limit > 0:
            event_ids = event_ids[-event_limit:]
        else:
            event_ids = []
        action_result = self.runtime.action_results.get(action_id or "", {})
        for output in output_buffer:
            output.extra_fields.update(
                {
                    "graph_version_before": graph_version_before,
                    "graph_version_after": self.runtime.version,
                    "action_id": action_id,
                    "activation_event_ids": event_ids,
                    "evidence_proposal_id": action_result.get("proposal_id"),
                    "accepted_node_count": action_result.get(
                        "accepted_node_count", len(self.runtime.evidence_graph.nodes)
                    ),
                    "rejected_node_count": 0,
                    "read_evidence_token_count": 0,
                    "graph_tool_phase": get_current_phase(),
                }
            )
        return output_buffer, answer_text, total_turn_list

    async def worker_call(
        self,
        worker_request: ToolRequest,
        main_task: str,
        sub_traj_id: int,
    ):
        """Run a worker with the action scope inherited by all child tasks."""

        action_id = str(worker_request.arguments.get("action_id", ""))
        if not action_id:
            raise ValueError("Graph worker request is missing action_id")
        if action_id in {"action:plan_task", "action:initial_frontier"}:
            raise ValueError(
                f"Graph action {action_id!r} is system-owned and cannot be delegated"
            )
        action = self.runtime.activation_dag.actions.get(action_id)
        if action is None or action.state != ActionState.READY:
            raise ValueError(f"Graph action {action_id!r} is not ready")
        self.runtime.mark_action_running(action_id, owner_sub_traj=sub_traj_id)
        action_token = self.runtime.action_context_token(action_id)
        sub_traj_token = self.runtime.sub_traj_context_token(sub_traj_id)
        try:
            return await super().worker_call(worker_request, main_task, sub_traj_id)
        finally:
            self.runtime.reset_sub_traj_context(sub_traj_token)
            self.runtime.reset_action_context(action_token)

    async def _finalize_trajectory(
        self,
        *,
        role: str,
        response_text: str,
        turn_idx: int,
        total_turn_list: list,
        conv_id: str,
    ):
        """Audit and render planner output; never accept free-form direct text."""

        if role != "planner":
            return await super()._finalize_trajectory(
                role=role,
                response_text=response_text,
                turn_idx=turn_idx,
                total_turn_list=total_turn_list,
                conv_id=conv_id,
            )
        audit = (
            audit_item(self.runtime)
            if self.runtime.finish_requested
            else AuditResult(
                passed=False,
                status="INCOMPLETE",
                gap_report=GapReport(missing_fields=("propose_finish",)),
            )
        )
        self.runtime.last_audit = audit
        self.runtime.record_completion_audit(audit.passed)
        if audit.passed:
            answer_text = render(self.runtime)
        else:
            answer_text = ""
        total_turn_list.append(turn_idx + 1)
        self.release_affinity(conv_id)
        return answer_text, total_turn_list

    async def run_one_query(self, prompt_ids: list[int], *, answer):
        """Create, isolate, and dispose one graph runtime per rollout."""

        origin_question = self.tokenizer.decode(prompt_ids)
        answer_type = str(answer.get("answer_type", "table")).lower()
        if answer_type != "item":
            raise ValueError(
                "workflow=mas_graph Phase 1 supports only answer_type=item; "
                f"got {answer_type!r}"
            )
        runtime = GraphRuntime.bootstrap(
            question=origin_question,
            answer_type=answer_type,
            config=self.graph_config,
        )
        runtime.begin_turn()
        token = runtime.context_token()
        action_token = runtime.action_context_token(None)
        sub_traj_token = runtime.sub_traj_context_token(0)
        try:
            output = await super().run_one_query(prompt_ids, answer=answer)
            audit = (
                audit_item(runtime)
                if runtime.finish_requested
                else AuditResult(
                    passed=False,
                    status="INCOMPLETE",
                    gap_report=GapReport(missing_fields=("propose_finish",)),
                )
            )
            runtime.last_audit = audit
            runtime.record_completion_audit(audit.passed)
            output.extra_fields.update(
                {
                    "task_contract": runtime.contract,
                    "graph_summary": runtime.summary(),
                    "activation_summary": {
                        "events": sum(
                            len(queue) for queue in runtime.event_queues.values()
                        ),
                        "phase_history": list(runtime.phase_history),
                    },
                    "completion_audit": audit,
                    "render_fact_refs": audit.fact_refs,
                    "graph_metrics": {
                        "graph_version": runtime.version,
                        "accepted_nodes": len(runtime.evidence_graph.nodes),
                        "accepted_edges": len(runtime.evidence_graph.edges),
                        **graph_credit_metrics(runtime),
                    },
                }
            )
            return output
        finally:
            runtime.begin_turn()
            runtime.reset_sub_traj_context(sub_traj_token)
            runtime.reset_action_context(action_token)
            runtime.reset_context(token)
