# Copyright 2026 The RLinf Authors.

from rlinf.agents.wideseek_r2.graph_memory.scheduler import GraphScheduler
from rlinf.agents.wideseek_r2.graph_memory.schema import (
    ActionNode,
    ActionState,
    TaskContract,
    TaskPlanProposal,
)
from rlinf.agents.wideseek_r2.graph_memory.state import GraphRuntime


def test_scheduler_returns_priority_ordered_frontier():
    import asyncio

    async def run():
        runtime = GraphRuntime.bootstrap(question="q", answer_type="item")
        await runtime.submit_task_plan(
            TaskPlanProposal(
                contract=TaskContract("contract", "q", "item"),
                actions=(
                    ActionNode("low", "low", priority=1),
                    ActionNode("high", "high", priority=10),
                ),
            )
        )
        scheduler = GraphScheduler(runtime)
        assert scheduler.ready_frontier()[0].action_id == "high"
        scheduler.start("high")
        scheduler.complete("high")
        assert runtime.activation_dag.actions["high"].state == ActionState.COMPLETED

    asyncio.run(run())
