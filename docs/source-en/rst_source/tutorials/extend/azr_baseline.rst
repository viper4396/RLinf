AZR Phase 0 Baseline Contract
=============================

The Absolute Zero Reasoner (AZR) migration starts from a frozen reproduction
contract. The contract targets the paper implementation at
``origin/paper@41ed983`` and Qwen2.5-Coder-3B. It is intentionally separate
from the current AZR ``master`` defaults and from the generic RLinf reasoning
configuration.

Frozen semantics
----------------

The contract fixes the following behavior:

* FSDP actor and vLLM rollout with tensor parallel size 2;
* train batch size 64, prompt length 6144, response length 8096;
* PROPOSE rollout ``n=1`` and difficulty SOLVE rollout ``n=8``;
* Reinforce++ advantage, PPO clipped actor loss, and no Critic;
* ``code_i``, ``code_o``, and ``code_f`` task types;
* difficulty trajectories are scored but excluded from the training batch.

The checked-in expansion is
``examples/reasoning/config/absolute_zero/baseline_config.yaml``.

Freeze and validate
-------------------

From the RLinf repository root, generate the manifest and deterministic fixed
examples with:

.. code-block:: bash

   python scripts/azr/freeze_baseline.py \
     --azr-root ~/AZR \
     --rlinf-root /path/to/clean/rlinf-a099ff4a-worktree \
     --output-dir examples/reasoning/config/absolute_zero

The command records SHA256 values for the two available seed datasets and
marks missing validation/model artifacts as blocking instead of silently
continuing. Supply the externally prepared files before a strict GPU run:

.. code-block:: bash

   python scripts/azr/freeze_baseline.py \
     --azr-root ~/AZR \
     --rlinf-root /path/to/clean/rlinf-a099ff4a-worktree \
     --validation-data /path/to/test_answer.parquet \
     --model-checkpoint /path/to/Qwen2.5-Coder-3B \
     --baseline-trace /path/to/baseline_trace.jsonl \
     --output-dir examples/reasoning/config/absolute_zero

Then gate the run with:

.. code-block:: bash

   python scripts/azr/freeze_baseline.py \
     --validate-only examples/reasoning/config/absolute_zero/baseline_manifest.yaml \
     --strict

The strict gate also requires a recorded 20--100 step AZR short run. The
per-step trace must retain proposal responses, difficulty and formal solve
responses, rewards, valid programs, and Program Pool size. A manifest with
missing external artifacts or without that trace remains pending by design.
The validator additionally checks that every proposal is linked to exactly
eight difficulty responses.

Artifacts
---------

* ``baseline_manifest.yaml`` records revisions, hashes, environment metadata,
  configuration, and gate status.
* ``fixed_examples.jsonl`` contains 50 deterministic examples: 20 ``code_i``,
  20 ``code_o``, and 10 ``code_f``.
* ``baseline_config.yaml`` is the explicit expansion of the paper
  ``coder3b.sh`` overrides.
