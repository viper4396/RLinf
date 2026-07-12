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

from rlinf.agents.searchr1.reference_runner import SearchR1ReferenceRunnerMixin
from rlinf.runners.agent_runner import AgentRunner


class Searchr1AgentRunner(SearchR1ReferenceRunnerMixin, AgentRunner):
    """Search-R1 training runner with isolated reward references."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_searchr1_reference_channel()
