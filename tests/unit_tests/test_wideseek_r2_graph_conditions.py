# Copyright 2026 The RLinf Authors.

import pytest

from rlinf.agents.wideseek_r2.graph_memory.conditions import (
    ConditionError,
    evaluate_condition,
    validate_condition,
)
from rlinf.agents.wideseek_r2.graph_memory.schema import ActionNode, GateNode
from rlinf.agents.wideseek_r2.graph_memory.state import GraphRuntime


def test_gate_dsl_rejects_arbitrary_expression():
    with pytest.raises(ConditionError, match="Unsupported Gate operator"):
        validate_condition({"op": "python", "expression": "__import__('os')"})


def test_gate_dsl_composes_whitelisted_conditions():
    runtime = GraphRuntime.bootstrap(question="q", answer_type="item")
    runtime.activation_dag.add_node(ActionNode("done", "done"))
    runtime.activation_dag.replace_node(
        ActionNode("done", "done", state=runtime.action_state_completed)
    )
    runtime.activation_dag.add_node(GateNode("gate", {"op": "true"}, satisfied=True))
    assert evaluate_condition(
        {
            "op": "all",
            "conditions": [
                {"op": "true"},
                {"op": "actions_completed", "action_ids": ["done"]},
            ],
        },
        runtime,
    )
