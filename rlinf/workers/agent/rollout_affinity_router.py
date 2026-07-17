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

"""Load-aware, sticky conversation -> rollout-worker router for KV-cache affinity.

This is the "upgraded" affinity policy (``rollout_kv_affinity_load_aware``). It
keeps every agent conversation pinned to a single SGLang worker for its whole
life (so the growing prompt prefix stays warm in that worker's radix/KV cache),
while placing *new* conversations on the least-loaded worker.

Load is **turn-weighted**: an active conversation counts as the number of turns
it has generated so far. Because rollout here is decode-dominated and runs with
a warm prefix cache, per-turn GPU cost is comparable across turns, so turn count
is a good unit-free proxy for a conversation's accumulated / ongoing load. A
worker hosting long-lived (many-turn) conversations therefore looks heavier and
repels new work, which equalizes *remaining* work across workers and shrinks the
straggler tail (planner blocks on its slowest worker sub-trajectory).

A single Ray actor holds the state, so all agent-loop worker processes share one
global, coordinated view (this is what prevents independent producers from all
piling new conversations onto the same "empty-looking" worker).
"""

import zlib
from typing import Optional

import ray
from omegaconf import DictConfig


@ray.remote(num_cpus=0)
class RolloutAffinityRouter:
    """Singleton actor mapping conversations to rollout-worker ranks.

    Args:
        num_workers: Number of main SGLang rollout DP workers (``rollout_dp_size``).
        long_tail_beta: Extra weight each additional turn adds to a conversation.
            ``1.0`` means "balance total in-flight turns"; ``0.0`` degrades to
            balancing the raw number of active conversations. Larger values push
            new work off workers that host long conversations more aggressively.
    """

    def __init__(self, num_workers: int, long_tail_beta: float = 1.0):
        self.n = max(1, int(num_workers))
        self.beta = float(long_tail_beta)
        self.load = [0.0] * self.n  # per-worker aggregate weight
        self.rank_of: dict[str, int] = {}  # conv_id -> assigned rank (sticky)
        self.weight_of: dict[str, float] = {}  # conv_id -> its current weight

    def assign(self, conv_id: str) -> int:
        """Return the sticky rank for ``conv_id``, assigning it on first sight.

        New conversations go to the least-loaded worker; ties break deterministic-
        ally starting from the conversation's hash position so equal-load workers
        still spread out.
        """
        rank = self.rank_of.get(conv_id)
        if rank is None:
            start = zlib.crc32(conv_id.encode()) % self.n
            rank = min(
                range(self.n),
                key=lambda r: (self.load[r], (r - start) % self.n),
            )
            self.rank_of[conv_id] = rank
            self.weight_of[conv_id] = 1.0  # first turn
            self.load[rank] += 1.0
        return rank

    def bump(self, conv_id: str) -> None:
        """Record that ``conv_id`` generated one more turn (grows its weight)."""
        rank = self.rank_of.get(conv_id)
        if rank is None:
            return
        self.weight_of[conv_id] += self.beta
        self.load[rank] += self.beta

    def release(self, conv_id: str) -> None:
        """Free a finished conversation's weight from its worker."""
        rank = self.rank_of.pop(conv_id, None)
        weight = self.weight_of.pop(conv_id, 0.0)
        if rank is not None:
            self.load[rank] = max(0.0, self.load[rank] - weight)

    def snapshot(self) -> dict:
        """Return current per-worker load and active-conversation count (debug)."""
        return {
            "load": list(self.load),
            "active_conversations": len(self.rank_of),
        }


def maybe_create_affinity_router(
    cfg: DictConfig, num_workers: int
) -> Optional["ray.actor.ActorHandle"]:
    """Create the router actor iff the load-aware affinity policy is enabled.

    Returns ``None`` for the disabled / simple-hash / shared-FIFO paths, which
    need no router.
    """
    if not cfg.rollout.get("enable_rollout_kv_affinity", False):
        return None
    if not cfg.rollout.get("rollout_kv_affinity_load_aware", False):
        return None
    beta = cfg.rollout.get("rollout_kv_affinity_long_tail_beta", 1.0)
    return RolloutAffinityRouter.remote(num_workers, beta)
