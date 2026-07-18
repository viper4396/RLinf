# Copyright 2025 The RLinf Authors.
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

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.agents.searchr1.eval_runner import Searchr1AgentEvalRunner as AgentEvalRunner
from rlinf.agents.searchr1.reward_worker import SearchR1RewardWorker
from rlinf.agents.searchr1.search_tool_worker import SearchToolWorker
from rlinf.agents.searchr1.searchr1_agent_loop import Searchr1AgentLoopWorker
from rlinf.config import validate_cfg
from rlinf.data.datasets import create_rl_dataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.scheduler import Cluster, NodePlacementStrategy, PackedPlacementStrategy
from rlinf.utils.placement import ModelParallelEvalComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.agent.tool_worker import ToolWorkerInfo
from rlinf.workers.rollout.utils import get_rollout_backend_worker

"""Script to start Search-R1 evaluation"""
mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelEvalComponentPlacement(cfg, cluster)

    # Generator group
    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_placement_strategy = component_placement.get_strategy("rollout")
    rollout_group = rollout_worker_cls.create_group(
        cfg, component_placement, weight_reload=None
    ).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=rollout_placement_strategy,
    )

    solid_rollouts = {}
    teacher_cfg = cfg.get("teacher_planner", {})
    if teacher_cfg.get("enabled", False) and not teacher_cfg.get("cache_only", False):
        if teacher_cfg.rollout_backend != "sglang":
            raise ValueError(
                "Search-R1 teacher_planner currently requires the SGLang "
                "serverless rollout backend"
            )
        if teacher_cfg.rollout_backend != cfg.rollout.rollout_backend:
            raise ValueError(
                "teacher_planner and policy rollout must use the same backend"
            )
        teacher_hardware = sorted(
            component_placement.get_hardware_ranks("teacher_planner")
        )
        if teacher_hardware != list(
            range(teacher_hardware[0], teacher_hardware[-1] + 1)
        ):
            raise ValueError("teacher_planner hardware ranks must be contiguous")
        teacher_parallel_size = int(teacher_cfg.get("tensor_parallel_size", 1)) * int(
            teacher_cfg.get("pipeline_parallel_size", 1)
        )
        teacher_placement = PackedPlacementStrategy(
            teacher_hardware[0],
            teacher_hardware[-1],
            num_hardware_per_process=teacher_parallel_size,
        )
        solid_rollouts["teacher_planner"] = rollout_worker_cls.create_group(
            cfg,
            component_placement,
            weight_reload=None,
            config_rollout=teacher_cfg,
        ).launch(
            cluster,
            name=teacher_cfg.group_name,
            placement_strategy=teacher_placement,
        )

    # AgentLoop group.
    agentloop_placement_strategy = NodePlacementStrategy(
        [
            placement.cluster_node_rank
            for placement in rollout_placement_strategy.get_placement(cluster)
        ]
    )
    assert (
        len(agentloop_placement_strategy._node_ranks)
        == component_placement.rollout_dp_size
    ), "agentloop worker num now should be equal to rollout dp size"
    agentloop_group = Searchr1AgentLoopWorker.create_group(
        cfg, component_placement
    ).launch(
        cluster,
        name=cfg.agentloop.group_name,
        placement_strategy=agentloop_placement_strategy,
    )

    # Dataset
    tokenizer = hf_tokenizer(cfg.rollout.model.model_path)
    train_ds, val_ds = create_rl_dataset(cfg, tokenizer)

    # Tool workers group
    singleton_tool_placement = NodePlacementStrategy([0])
    reward_group = SearchR1RewardWorker.create_group(cfg).launch(
        cluster,
        name=cfg.reward.group_name,
        placement_strategy=singleton_tool_placement,
    )
    tool_workers = {
        SearchToolWorker.create_group(cfg).launch(
            cluster, name="search", placement_strategy=singleton_tool_placement
        ): ToolWorkerInfo(tool_names=["search"], has_session=False),
    }

    runner = AgentEvalRunner(
        cfg=cfg,
        placement=component_placement,
        val_dataset=val_ds,
        rollout=rollout_group,
        reward=reward_group,
        agent_loop=agentloop_group,
        tool_workers=tool_workers,
        solid_rollouts=solid_rollouts,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
