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

from uuid import uuid4

from rlinf.data.io_struct import RolloutRequest
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import split_list


class SearchR1ReferenceRunnerMixin:
    """Keep Search-R1 ground truth on a reward-only side channel."""

    def _init_searchr1_reference_channel(self) -> None:
        if self.reward is None:
            raise ValueError("Search-R1 requires its dynamic reward worker")
        self.reward_reference_channel = Channel.create("SearchR1RewardReference")

    def _put_batch(self, batch: dict, split_size=None) -> None:
        """Send prompts to rollout and references directly to the reward worker."""
        prompt_ids = batch["prompt"].tolist()
        lengths = batch["length"].tolist()
        answers = batch["answer"]
        image_data = batch["image_data"]
        multi_modal_inputs = batch["multi_modal_inputs"]
        prompt_ids = [ids[-prompt_len:] for ids, prompt_len in zip(prompt_ids, lengths)]
        reference_ids = [uuid4().hex for _ in prompt_ids]

        if split_size is None:
            split_size = self.component_placement.rollout_dp_size

        split_inputs = zip(
            split_list(prompt_ids, split_size, enforce_divisible_batch=False),
            split_list(answers, split_size, enforce_divisible_batch=False),
            split_list(image_data, split_size, enforce_divisible_batch=False),
            split_list(multi_modal_inputs, split_size, enforce_divisible_batch=False),
            split_list(reference_ids, split_size, enforce_divisible_batch=False),
        )
        for (
            input_ids,
            answer_batch,
            image_batch,
            multi_modal_batch,
            reference_id_batch,
        ) in split_inputs:
            self.reward_reference_channel.put(
                {
                    "reference_ids": reference_id_batch,
                    "answers": answer_batch,
                },
                async_op=True,
            )
            # MultiAgentLoopWorker calls this field `answers`, but Search-R1 only
            # receives opaque IDs. The corresponding GT never enters agent-loop
            # memory or any model prompt.
            request = RolloutRequest(
                n=self.cfg.algorithm.group_size,
                input_ids=input_ids,
                answers=reference_id_batch,
                image_data=image_batch,
                multi_modal_inputs=multi_modal_batch,
            )
            self.dataloader_channel.put(request, async_op=True)

    def _get_reward_compute_kwargs(self, batch: dict) -> dict:
        return {
            "reference_channel": self.reward_reference_channel,
            "total_batch_size": len(batch["answer"]),
        }
