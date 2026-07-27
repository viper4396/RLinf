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

"""Deterministic frontier and activation-event scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlinf.agents.wideseek_r2.graph_memory.schema import ActionState, ActivationEvent
from rlinf.agents.wideseek_r2.graph_memory.state import GraphRuntime, GraphStateError


@dataclass(frozen=True)
class ReadyAction:
    """Small scheduler view used by planner summaries and tests."""

    action_id: str
    objective: str
    priority: int
    state: ActionState


class GraphScheduler:
    """Schedule ready actions without embedding LLM semantics in the system."""

    def __init__(self, runtime: GraphRuntime):
        self.runtime = runtime

    def reevaluate(self) -> list[dict[str, Any]]:
        """Recompute Gate/action transitions after a graph change."""

        return self.runtime.evaluate_activation()

    def ready_frontier(self, *, limit: int | None = None) -> list[ReadyAction]:
        actions = [
            ReadyAction(
                action_id=action.action_id,
                objective=action.objective,
                priority=action.priority,
                state=action.state,
            )
            for action in self.runtime.activation_dag.actions.values()
            if action.state == ActionState.READY
        ]
        actions.sort(key=lambda item: (-item.priority, item.action_id))
        return actions[:limit] if limit is not None else actions

    def start(self, action_id: str, *, owner_sub_traj: int | None = None) -> None:
        """Move one ready action to running and consume its ready packet."""

        self.assert_capacity()
        self.runtime.mark_action_running(action_id, owner_sub_traj=owner_sub_traj)
        self.runtime.pending_events(action_id, consume=True)

    def complete(
        self,
        action_id: str,
        *,
        summary: str = "",
        partition: str | None = None,
    ) -> list[dict[str, Any]]:
        """Mark an action completed and expose newly ready downstream work."""

        return self.runtime.mark_action_completed(
            action_id,
            status="completed",
            summary=summary,
            partition=partition,
        )

    def fail(self, action_id: str, *, reason: str, retry: bool = False) -> None:
        """Record a failed or blocked action deterministically."""

        status = "blocked" if retry else "failed"
        self.runtime.mark_action_completed(action_id, status=status, summary=reason)

    def events_for_action(
        self, action_id: str, *, consume: bool = False
    ) -> list[ActivationEvent]:
        return self.runtime.pending_events(action_id, consume=consume)

    def choose_action(self, action_ids: list[str] | None = None) -> ReadyAction | None:
        """Return the highest-priority ready action from an optional subset."""

        frontier = self.ready_frontier()
        if action_ids is not None:
            allowed = set(action_ids)
            frontier = [item for item in frontier if item.action_id in allowed]
        return frontier[0] if frontier else None

    def assert_capacity(self) -> None:
        running = sum(
            action.state == ActionState.RUNNING
            for action in self.runtime.activation_dag.actions.values()
        )
        if running >= self.runtime.config.max_concurrent_actions:
            raise GraphStateError("Maximum concurrent graph actions exceeded")
