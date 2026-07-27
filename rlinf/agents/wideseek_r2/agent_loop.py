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

from rlinf.agents.wideseek_r2.wideseek_r2 import WideSeekR2AgentLoopWorker


def get_wideseek_r2_agent_loop_cls(
    workflow: str,
) -> type[WideSeekR2AgentLoopWorker]:
    """Return the agent-loop class for a WideSeek-R2 workflow.

    Existing workflows intentionally fall back to the current worker so this
    factory is behavior-preserving for `mas`, `sa`, and any legacy value.  The
    graph workflow is imported lazily to keep the current path independent of
    the graph-memory package while it is being developed.

    Args:
        workflow: Configured WideSeek-R2 workflow name.

    Returns:
        Agent-loop worker class for the requested workflow.
    """
    if str(workflow).lower() == "mas_graph":
        from rlinf.agents.wideseek_r2.graph_memory.agent_loop import (
            WideSeekR2GraphAgentLoopWorker,
        )

        return WideSeekR2GraphAgentLoopWorker
    return WideSeekR2AgentLoopWorker
