# Plan — Create `wideseek_r2` from `wideseek_r1` and Refactor It

## Goal Description

Create a new agent named `wideseek_r2` by making a complete copy of the existing `wideseek_r1` implementation, then refactor **only the `wideseek_r2` copy** to be simpler and clearer. The original `wideseek_r1` (and every other agent) must remain functionally unchanged.

The work has two parts:

- **Part 1 — Copy & rename.** Duplicate the `wideseek_r1` example directory, agent package, dataset module, and the `compute_grpo_dynamic_advantages` advantage (copied under the new name `gigpo`), then rename every identifier, import, class, log string, config value, metric prefix, file, and directory from `wideseek_r1` to `wideseek_r2`. Register the new dataset type, tool-call parser, and advantage so the copy is runnable.
- **Part 2 — Refactor (`wideseek_r2` only).** English-only prompts; make few-shot inclusion a config option; stop hardcoding tool-call/sub-agent limits; remove the complex credit-assignment logic so the reward is computed from only the final answer score plus the format reward; simplify `run_one_query_role` and introduce an `<answer>…</answer>` worker return contract; reorganize utility functions into dedicated modules; and rename `is_markdown` to `answer_mode` (values `markdown` and `boxed`).

Two refactor items in the draft interact with shared infrastructure and were resolved with the user (see **Pending User Decisions**): the `not_training` filter lives in a shared worker file, and the eval acceptance commands reference the `wideseek_r1` path even though the refactor lives in `wideseek_r2`.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive tests (expected to PASS when the criterion is met) and negative tests (expected to FAIL / be rejected when the system is working correctly). Reward-threshold criteria are **directional** per DEC-3 (observed in a smoke run on a healthy setup, not deterministic gates).

- AC-1: `wideseek_r2` exists as a complete, runnable copy and `wideseek_r1` is unchanged.
  - Positive Tests (expected to PASS):
    - `python -c "import rlinf.agents.wideseek_r2.wideseek_r2"` imports cleanly, exposing `WideSeekR2AgentLoopWorker`.
    - `git diff --name-only` shows **no** modifications under `rlinf/agents/wideseek_r1/`, `examples/agent/wideseek_r1/`, or `rlinf/data/datasets/wideseek_r1.py`.
    - `rlinf/data/datasets/wideseek_r2.py` defines `WideSeekR2Dataset`, registered for `data.type: wideseek_r2`.
  - Negative Tests (expected to FAIL):
    - `grep -rn "wideseek_r1" rlinf/agents/wideseek_r2 examples/agent/wideseek_r2 rlinf/data/datasets/wideseek_r2.py` returns **no** lingering `wideseek_r1` references (no leftover imports, class names, metric prefixes, or paths).
    - Importing `wideseek_r2` must not import any `rlinf.agents.wideseek_r1.*` module.
  - AC-1.1: All copied entry points are renamed.
    - Positive: `examples/agent/wideseek_r2/train.py` and `eval.py` import `WideSeekR2AgentLoopWorker` and `WideSeekR2ToolWorker`; `tools.py`, `eval_runner.py` use r2 class/log names.
    - Negative: A run of `examples/agent/wideseek_r2/run_train.sh` must not execute `examples/agent/wideseek_r1/train.py` or instantiate any `WideSeekR1*` class.

- AC-2: `algorithm.adv_type: gigpo` works end-to-end.
  - Positive Tests: `compute_gigpo_advantages` is registered via `@register_advantage("gigpo")`; the advantage preprocessing recognizes `gigpo` identically to `grpo_dynamic`; config validation accepts `gigpo` (and still requires `group_size > 1`); a training step with `adv_type: gigpo` computes advantages without error.
  - Negative Tests: Running with `adv_type: gigpo` must **not** raise `Unsupported adv_type gigpo`; validation must still **reject** `gigpo` with `group_size <= 1`.

- AC-3: Prompts are English-only, few-shot is configurable, and tool limits are not hardcoded (`wideseek_r2` only).
  - Positive Tests:
    - `grep -rnP "_ZH\b|enable_zh|[\x{4e00}-\x{9fff}]" rlinf/agents/wideseek_r2 rlinf/data/datasets/wideseek_r2.py` finds no `*_ZH` symbols, no `enable_zh`, and no CJK characters in r2 prompt/dataset code.
    - Setting `agentloop.add_few_shot: false` produces the NOSHOT system prompt; `true` produces the few-shot prompt — independent of `answer_mode`.
    - With `max_workers_per_planner: 7` and `max_toolcall_per_worker: 3`, the generated tool descriptions and the planner few-shot narrative state 7 sub-agents and 3 tool instances.
  - Negative Tests:
    - No code path selects a Chinese prompt for any `language`/record value (the `language == "zh"` branches are gone).
    - Tool descriptions must **not** contain the literal words “ten”/“five” (or a hardcoded “10 parallel subtasks”) independent of config.

- AC-4: Reward uses only the final answer score plus the format reward (`wideseek_r2` only).
  - Positive Tests:
    - `credit_assignment` computes `reward_score` from only `llm_reward` and `format_reward` and returns `reward_score`; no `train_buffer`, search credit, length penalty, or over-context penalty remains.
    - `wideseek_r2` configs contain no `call_search_reward`, `length_limit`, `max_length_limit`, `length_penalty`, or `over_context_penalty` keys.
    - The caller assigns `reward_score` to all turns, sets `not_training=False` on every turn, and records `final_answer_format = (final_answer_extract is not None and format is True)` as a trajectory metric.
  - Negative Tests:
    - Setting a removed key (e.g. `length_penalty`) in a `wideseek_r2` config must have **no effect** on the reward (the code no longer reads it).
    - `rlinf/workers/agent/agent_loop.py` must remain unchanged; `git diff` shows it untouched.

- AC-5: `run_one_query_role` is simplified and the `<answer>…</answer>` worker contract is implemented (`wideseek_r2` only).
  - Positive Tests:
    - `run_one_query_role` no longer returns `task_failed` or `succ_end`; `_mark_role_failed_turns`, `context_failed`, and `tool_response_failed` are removed; all call sites are updated to the new return shape.
    - The worker system prompt instructs the worker to return its final answer inside `<answer>…</answer>`.
    - When a worker response contains `<answer>X</answer>`, `extract_final_answer(text, mode="tag")` returns `X`, and the planner receives `get_planner_subtask_result_message(...)` with `X` as the summary.
    - When a worker response contains no `<answer>` tag, extraction returns `None` and the planner receives `get_planner_subtask_failed_message(...)`.
  - Negative Tests:
    - The old `response_text.split("</think>")[-1].split("<|im_end|>")[0]` worker-summary path must be gone.
    - A worker with a malformed/absent `<answer>` tag must **not** be routed to `get_planner_subtask_result_message`.

- AC-6: Utility functions are reorganized (`wideseek_r2` only).
  - Positive Tests:
    - `rlinf/agents/wideseek_r2/utils/utils.py` contains `_build_tool_call_info`, a standalone `_set_max_turns` that takes the `agentloop` config as a parameter, and a standalone function holding the extracted valid-turn-count / per-turn extra-field logic (the block the draft calls out as lines 719–752).
    - `_build_message_history_and_tools` lives in `utils/prompt_utils.py`; `get_rollout_metrics` lives in `utils/metrics.py`.
    - The agent module imports these from their new locations and runs.
  - Negative Tests:
    - Moving these helpers must not change behavior: a rollout produces the same metric keys and tool-call-info structure as before the move.
    - `gen_extra_fields` must **not** be turned into a free function that drops its `super().gen_extra_fields(...)` call or its use of `self.extra_keys_turn` (it stays a thin method wrapper).

- AC-7: `is_markdown` is renamed to `answer_mode` across `wideseek_r2` (`markdown` / `boxed`).
  - Positive Tests:
    - `wideseek_r2` configs and code use `answer_mode` (values `markdown`/`boxed`); the dataset emits `answer_mode`; prompt selection, `get_final_reward_score`, `extract_final_answer` mode, and eval aggregation all branch on `answer_mode`.
    - `answer_mode: markdown` drives markdown extraction + item-level F1; `answer_mode: boxed` drives boxed extraction + LLM-judge scoring.
    - For backward compatibility (DEC-4), a hybrid data record carrying a boolean `is_markdown` is normalized to `answer_mode` (`True → markdown`, `False → boxed`).
  - Negative Tests:
    - No `wideseek_r2` config key or code logic named `is_markdown` remains, **except** the dataset reader’s accepted legacy input key (normalized immediately to `answer_mode`).
    - An unsupported `answer_mode` value is rejected (not silently treated as boxed).

- AC-8: Training smoke run (directional, per DEC-3).
  - Positive Tests: `bash examples/agent/wideseek_r2/run_train.sh` completes at least one training step with correct metric logging under the `wideseek_r2/...` namespace; the observed one-step training reward is in the expected range (≈ ≥ 0.3 on a healthy setup).
  - Negative Tests: The run must not crash on missing registration (dataset/parser/advantage), `KeyError: 'not_training'`, or unsupported-`adv_type` errors; metric logging must not be empty/malformed.

- AC-9: Eval smoke run for both modes (directional, per DEC-3; eval target is `wideseek_r2` per DEC-2).
  - Positive Tests: `bash examples/agent/wideseek_r2/run_eval.sh eval_qwen3_width` (markdown) and `bash examples/agent/wideseek_r2/run_eval.sh eval_qwen3_qa` (boxed) finish normally with correct metric logging; observed LLM reward is in the expected range (≈ ≥ 0.3) in both.
  - Negative Tests: Eval must not error on metric aggregation for either mode; markdown-only metrics must not be computed for the boxed config and vice versa.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A complete, runnable `wideseek_r2` agent that is a faithful copy of `wideseek_r1` with all seven refactors applied; the new dataset type, `wideseek_r2-qwen` tool-call parser, and `gigpo` advantage registered (with additive preprocessing + validation entries); `wideseek_r2` train/eval configs cleaned of the removed reward-shaping keys and switched to `answer_mode`, `gigpo`, and the r2 parser; matching `wideseek_r2` run scripts and entry points; optional `wideseek_r2` e2e test configs; small unit tests for `answer_mode` normalization and `<answer>` extraction; and a successful one-step training smoke run plus markdown and boxed eval smoke runs. The original `wideseek_r1` and all shared files except the additive `gigpo` registrations remain unchanged.

### Lower Bound (Minimum Acceptable Scope)
A runnable `wideseek_r2` that satisfies AC-1 through AC-9: the copy is renamed and registered; all seven refactors are implemented in `wideseek_r2` only; `gigpo` is wired end-to-end; `is_markdown` is replaced by `answer_mode` with legacy normalization; and the training + both eval smoke runs execute end-to-end with correct metric logging (reward in the expected directional range). Optional e2e config scaffolding and exhaustive unit tests may be minimal, but `answer_mode` normalization and `<answer>` extraction should have at least basic coverage.

### Allowed Choices
- Can use: the existing RLinf registries (`register_advantage`, `register_toolcall_parser`, `dataset_type_map`); reuse of `extract_final_answer(mode="tag")` for the worker contract; a thin-wrapper `gen_extra_fields` that delegates to new utility helpers; copying the parser/tool classes under new `WideSeekR2*` names; additive edits to shared `advantages.py` / `algorithms/utils.py` / `config.py` strictly for `gigpo`.
- Can use: `agentloop.add_few_shot` (default `true`) as the config owner for few-shot (DEC-5).
- Cannot use: any edit to `rlinf/workers/agent/agent_loop.py` (DEC-1); any modification to `wideseek_r1` files or other agents’ behavior; deletion of the shared `not_training` filter; Chinese prompts or `*_ZH` symbols in `wideseek_r2`; hardcoded tool/sub-agent limits; the legacy reward-shaping logic (search credit, length/over-context penalties, `train_buffer`).

> **Note on Deterministic Designs**: The draft is highly prescriptive (exact files, exact deletions, exact return contracts). Where the draft fixes a choice, the bounds converge to that choice. The reward-threshold criteria are the main non-deterministic part and are explicitly directional per DEC-3.

## Feasibility Hints and Suggestions

> **Note**: Reference only — one possible path, not prescriptive.

### Conceptual Approach
1. Copy the four sources to `wideseek_r2` (exclude `__pycache__`). Mechanically rename `wideseek_r1 → wideseek_r2` and `WideSeekR1* → WideSeekR2*` across the copy (package imports, class names, log strings, metric prefixes, dataset class, entry-point imports, run-script paths/defaults, config `data.type` and `toolcall_parser`).
2. Register the new pieces: add `WideSeekR2Dataset` to `dataset_type_map` (`rlinf/data/datasets/__init__.py`); add `@register_toolcall_parser("wideseek_r2-qwen")` (a `WideSeekR2QwenToolCallParser` copy); add `@register_advantage("gigpo") compute_gigpo_advantages` (identical body to `compute_grpo_dynamic_advantages`) plus a `gigpo` branch in the advantage preprocessing and a `gigpo` entry in the `config.py` group-size validation tuple.
3. Apply the refactors in `wideseek_r2` only:
   - Delete all ZH constants/functions/branches in `utils/prompt.py`, `utils/prompt_utils.py`, `utils/tool_description.py`, and the `enable_zh` + language detection in the dataset.
   - Introduce `agentloop.add_few_shot` (default `true`) and thread it through the prompt builders, replacing `add_few_shot = is_markdown`.
   - Parametrize tool-description and few-shot limits from `max_workers_per_planner` / `max_toolcall_per_worker` (numeric).
   - Rewrite `credit_assignment(agentloop_config, llm_reward, answer_format) -> reward_score` where `reward_score = llm_reward + (format_reward if answer_format else 0.0)`; rewrite the caller to assign that score to all turns, set `not_training=False` on every turn, and record `final_answer_format` as a trajectory metric; remove `train_buffer` handling. Leave `agent_loop.py` untouched.
   - Simplify `run_one_query_role` (drop `task_failed`/`succ_end`, remove `_mark_role_failed_turns`/`context_failed`/`tool_response_failed`), and replace the worker summary split with `extract_final_answer(mode="tag")`; route planner result vs. failed message on extraction success; add the `<answer>` instruction to the worker system prompt.
   - Move helpers into `utils/utils.py`, `utils/prompt_utils.py`, `utils/metrics.py`; keep `gen_extra_fields` a thin wrapper.
   - Rename `is_markdown → answer_mode` end to end, with legacy per-record `is_markdown` normalized in the dataset.
4. Clean the `wideseek_r2` configs (remove shaping keys; set `answer_mode`, `adv_type: gigpo`, `toolcall_parser: wideseek_r2-qwen`, `data.type: wideseek_r2`).
5. Smoke-test: one training step, then markdown and boxed eval.

### Relevant References
- `rlinf/agents/wideseek_r1/wideseek_r1.py` — `WideSeekR1AgentLoopWorker`; `run_one_query_role`, `worker_call`, `run_one_query`, `_mark_role_failed_turns`, `gen_extra_fields`, `get_rollout_metrics`, `_build_message_history_and_tools`, `_set_max_turns`, `_build_tool_call_info`, the worker-summary split, and the valid-turn-count block.
- `rlinf/agents/wideseek_r1/utils/reward.py` — `credit_assignment`, `get_final_reward_score`, `evaluate_markdown`, `verify_answer_with_llm_judge`, `extract_final_answer` (supports `tag`/`markdown`/`boxed`), `LLM_JUDGE_PROMPT` import.
- `rlinf/agents/wideseek_r1/utils/prompt.py`, `prompt_utils.py`, `tool_description.py` — ZH symbols, `add_few_shot = is_markdown`, hardcoded limits, `SYSTEM_PROMPT_WORKER`, `get_planner_subtask_result_message` / `get_planner_subtask_failed_message`.
- `rlinf/agents/wideseek_r1/tools.py`, `eval_runner.py` — `WideSeekR1ToolWorker`, `WebPageCache`, `is_markdown` read in eval aggregation, r1-specific log/class names.
- `rlinf/data/datasets/wideseek_r1.py` + `rlinf/data/datasets/__init__.py` — `WideSeekR1Dataset`, `enable_zh`, per-record `is_markdown`/`is_hybrid`, `dataset_type_map`.
- `rlinf/algorithms/advantages.py`, `registry.py`, `utils.py`, `toolcall_parsers.py`, `rlinf/config.py` — advantage/parser registries, `adv_type` preprocessing + validation.
- `examples/agent/wideseek_r1/config/` (`base_train.yaml`, `base_eval.yaml`, `train_qwen3_hybrid.yaml`, `eval_qwen3_qa.yaml`, `eval_qwen3_width.yaml`), `run_train.sh`, `run_eval.sh`, `train.py`, `eval.py`.
- `tests/e2e_tests/agent/wideseek/` — e2e train/eval configs and run scripts to mirror for `wideseek_r2`.

## Dependencies and Sequence

### Milestones
1. **Copy & register the runnable `wideseek_r2` skeleton.**
   - Phase A: Copy the four sources to `wideseek_r2` (exclude `__pycache__`).
   - Phase B: Rename all identifiers/imports/classes/logs/metric-prefixes/paths and config `data.type`/`toolcall_parser`.
   - Phase C: Register dataset, `wideseek_r2-qwen` parser, and `gigpo` advantage (+ preprocessing + validation). After this milestone, an unrefactored `wideseek_r2` should import and (in principle) run.
2. **Refactor `wideseek_r2` (depends on Milestone 1).**
   - Phase A: English-only deletions (prompts + dataset).
   - Phase B: `add_few_shot` config + non-hardcoded limits.
   - Phase C: Reward simplification + caller rewrite (+ `not_training=False` on all turns).
   - Phase D: `run_one_query_role` simplification + `<answer>` worker contract.
   - Phase E: Utility reorganization.
   - Phase F: `is_markdown → answer_mode` rename + legacy normalization (touches dataset, prompts, reward, eval_runner, configs).
3. **Config cleanup, tests, and smoke validation (depends on Milestone 2).**
   - Step 1: Remove shaping keys; set `answer_mode`, `gigpo`, r2 parser/type in `wideseek_r2` train/eval configs (+ optional e2e configs).
   - Step 2: Add unit coverage for `answer_mode` normalization and `<answer>` extraction.
   - Step 3: Run training smoke (one step) and markdown + boxed eval smokes; confirm metric logging and directional reward.
4. **Impact analysis & adversarial verification (depends on Milestones 2–3).**
   - Verify `gigpo` advantage math still aligns with `idx_to_traj`/loss scaling once `train_buffer` selection is gone; verify `wideseek_r1` and other agents are untouched.

Dependency notes: Milestone 2 Phase F depends on Phases A–E being stable (rename touches files changed earlier). The `gigpo` wiring (Milestone 1 Phase C) is independent of the prompt/reward refactors and can proceed in parallel with Milestone 2 Phases A–B. Config cleanup (Milestone 3) depends on the reward and `answer_mode` refactors.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Copy the four `wideseek_r1` sources to `wideseek_r2` (example dir, agent package, dataset module, `gigpo` advantage copy), excluding `__pycache__` | AC-1 | coding | - |
| task2 | Rename all identifiers/imports/class names (`WideSeekR2*`), log strings, metric prefixes, file/dir names, run-script paths/defaults, and config `data.type`/`toolcall_parser` across the copy (incl. `tools.py`, `eval_runner.py`, `train.py`, `eval.py`, `run_train.sh`, `run_eval.sh`) | AC-1, AC-1.1 | coding | task1 |
| task3 | Register `WideSeekR2Dataset` in `dataset_type_map`; register `wideseek_r2-qwen` parser; register `gigpo` advantage + add `gigpo` preprocessing branch + add `gigpo` to the group-size validation tuple | AC-1, AC-2 | coding | task1 |
| task4 | Delete all ZH/Chinese prompt content and `language=="zh"` branches in r2 prompts; delete `enable_zh` + language detection in the r2 dataset | AC-3 | coding | task2 |
| task5 | Add `agentloop.add_few_shot` (default `true`) and thread it through r2 prompt builders, replacing `add_few_shot = is_markdown` | AC-3 | coding | task2 |
| task6 | Parametrize tool-description and few-shot limits from `max_workers_per_planner` / `max_toolcall_per_worker` (numeric) in r2 | AC-3 | coding | task2 |
| task7 | Rewrite r2 `credit_assignment` to `reward = llm_reward + format_reward` and return `reward_score`; rewrite caller to assign score to all turns, set `not_training=False`, record `final_answer_format` metric; remove `train_buffer`/shaping; leave `agent_loop.py` untouched | AC-4 | coding | task2 |
| task8 | Simplify r2 `run_one_query_role` (drop `task_failed`/`succ_end`, remove `_mark_role_failed_turns`/`context_failed`/`tool_response_failed`, update call sites); implement `<answer>` worker contract via `extract_final_answer(mode="tag")` and result/failed routing; add `<answer>` instruction to worker system prompt | AC-5 | coding | task2 |
| task9 | Reorganize r2 utilities into `utils/utils.py`, `utils/prompt_utils.py`, `utils/metrics.py`; keep `gen_extra_fields` a thin wrapper | AC-6 | coding | task7, task8 |
| task10 | Rename `is_markdown → answer_mode` (`markdown`/`boxed`) end to end in r2 (dataset, prompts, reward, `eval_runner`, configs); normalize legacy per-record `is_markdown` | AC-7 | coding | task4, task5, task7, task8 |
| task11 | Clean r2 train/eval configs (remove shaping keys; set `answer_mode`, `adv_type: gigpo`, `toolcall_parser: wideseek_r2-qwen`, `data.type: wideseek_r2`); create r2 eval configs `eval_qwen3_width` (markdown) and `eval_qwen3_qa` (boxed); optional r2 e2e configs | AC-2, AC-4, AC-7, AC-9 | coding | task3, task7, task10 |
| task12 | Add unit tests for `answer_mode` normalization and worker `<answer>` extraction | AC-5, AC-7 | coding | task8, task10 |
| task13 | Run one-step training smoke and markdown + boxed eval smokes; confirm metric logging under `wideseek_r2/...` and directional reward | AC-8, AC-9 | coding | task11 |
| task14 | Analyze whether removing `train_buffer` turn-selection (all turns now trainable) keeps `gigpo` advantage math / `idx_to_traj` / loss scaling consistent, and confirm `wideseek_r1` + other agents are byte-for-byte unaffected | AC-1, AC-2, AC-4 | analyze | task7, task11 |
| task15 | Final adversarial review: grep that no `wideseek_r1` refs/imports/metric-prefixes remain in r2, no `is_markdown` logic remains (except legacy reader), and `agent_loop.py`/r1 are unchanged | AC-1, AC-4, AC-7 | analyze | task13 |

## Claude-Codex Deliberation

### Agreements
- Copying the four sources to `wideseek_r2` and refactoring only the copy is the correct overall shape.
- `agent_loop.py` must stay unchanged; its `not_training` skip is shared and only `wideseek_r1` sets the flag.
- `gigpo` needs three pieces: a registered advantage, a preprocessing branch, and a validation-tuple entry — not just a renamed copy.
- The new dataset type and a `wideseek_r2-qwen` tool-call parser must be registered; otherwise the copy fails to run.
- `gen_extra_fields` cannot become a pure free function (it calls `super()` and uses `self.extra_keys_turn`); it stays a thin wrapper.
- Hardcoded limits also appear in the planner few-shot text, not only in tool descriptions.
- Reward thresholds are environment-dependent and should be treated as directional smoke observations.

### Resolved Disagreements
- **Shared `not_training` filter (draft says delete it).** Claude/Codex: the file is shared and only `wideseek_r1` sets the flag; deleting breaks r1. Resolution (DEC-1, user-confirmed): leave `agent_loop.py` unchanged and have `wideseek_r2` set `not_training=False` on every turn so the filter is a no-op and all r2 turns train.
- **Omitting `not_training` would `KeyError`.** Codex noted the copied `gen_extra_fields` indexes `extra_fields["not_training"]` directly. Resolution: r2 explicitly sets `not_training=False` on all turns (chosen over a `.get` default) so reads remain valid and intent (“all turns train”) holds.
- **Rename surface was under-specified.** Resolution: the rename checklist explicitly includes `tools.py`, `eval_runner.py`, `train.py`, `eval.py`, and the run scripts (imports, class names, log strings, metric prefixes, paths).
- **`credit_assignment` contract.** Resolution: `credit_assignment(agentloop_config, llm_reward, answer_format) -> reward_score`; the caller assigns the score to all turns and records `final_answer_format` as a trajectory metric (matching the draft’s `answer_format = final_answer_extract is not None and format is True`).
- **Removing `task_failed` vs. planner routing.** Resolution: worker success is determined by `extract_final_answer(mode="tag") is not None`, which drives the result-vs-failed planner message, replacing the old `task_failed`-based routing.
- **AC-7 vs. legacy data.** Resolution: `is_markdown` is removed from all r2 config/logic except as an accepted legacy input key in the dataset reader, normalized immediately to `answer_mode` (DEC-4).
- **Hardcoded limits in few-shot text.** Resolution: parametrize the planner few-shot narrative limits too, not only `tool_description.py`.

### Convergence Status
- Final Status: `converged` (one convergence round; all Codex `REQUIRED_CHANGES` incorporated; no Claude↔Codex disagreements remain — residual items were genuine user decisions, now resolved).

## Pending User Decisions

All decisions were resolved with the user during planning (none remain `PENDING`).

- DEC-1: How `wideseek_r2` achieves “all turns train” given the shared `not_training` filter.
  - Claude Position: Leave `agent_loop.py` unchanged; r2 sets `not_training=False` on all turns.
  - Codex Position: Same — the shared file must not be edited; this is effectively forced by the code.
  - Tradeoff Summary: Honors both “r1 unchanged” and the draft’s intent without touching shared infrastructure. (Alternative: override `get_rollout_result` in the r2 worker.)
  - Decision Status: **Resolved — leave `agent_loop.py` unchanged; r2 sets `not_training=False` on every turn.**

- DEC-2: Which agent the eval acceptance commands validate (draft path says `wideseek_r1`).
  - Claude Position: Eval `wideseek_r2` (the refactor + `answer_mode` live there); treat the r1 path as a typo.
  - Codex Position: Genuine human decision (regression vs. new behavior).
  - Tradeoff Summary: Eval-r2 validates the refactored markdown/boxed modes; eval-r1 would only confirm no regression.
  - Decision Status: **Resolved — eval `wideseek_r2`; create r2 `eval_qwen3_width` (markdown) and `eval_qwen3_qa` (boxed) and run `examples/agent/wideseek_r2/run_eval.sh`.**

- DEC-3: Whether reward thresholds (train ≥ 0.3 one-step; eval LLM reward ≥ 0.3) are hard gates or directional.
  - Claude Position: Directional (depends on model/search/judge/data).
  - Codex Position: Treat as smoke-run observations, not deterministic pass/fail.
  - Tradeoff Summary: Hard gates can’t be deterministic given external services; directional keeps acceptance reproducible.
  - Decision Status: **Resolved — directional targets; acceptance is end-to-end run + correct metric logging, with ≈ ≥ 0.3 expected on a healthy setup.**

- DEC-4: Handling legacy per-record boolean `is_markdown` in hybrid data after the `answer_mode` rename.
  - Claude Position: Normalize legacy `is_markdown` (`True → markdown`, `False → boxed`) in the r2 dataset reader.
  - Codex Position: Effectively forced if existing hybrid data must keep working.
  - Tradeoff Summary: Backward-compatible with existing jsonl; avoids regenerating data. (Alternative: require regenerated data using `answer_mode`.)
  - Decision Status: **Resolved — support legacy `is_markdown` as an input key, normalized to `answer_mode`.**

- DEC-5: Where `add_few_shot` lives and its default.
  - Claude Position: `agentloop.add_few_shot`, default `true`.
  - Codex Position: Genuine human decision (location + default).
  - Tradeoff Summary: Owning it under `agentloop` decouples few-shot from `answer_mode`; default `true` keeps few-shot on.
  - Decision Status: **Resolved — `agentloop.add_few_shot`, default `true`.**

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as “AC-”, “Milestone”, “Phase”, “Step”, “DEC-”, or similar workflow markers. These belong to this plan document only.
- Use descriptive, domain-appropriate names in code (e.g. `answer_mode`, `add_few_shot`, `compute_gigpo_advantages`, `WideSeekR2AgentLoopWorker`).
- Follow the repo conventions in AGENTS.md/CONTRIBUTING.md: Google Python style, Ruff lint/format, type hints + docstrings on public APIs, static config YAML (no computed fields), worker logging via `self.log_*`, Conventional Commits with `Signed-off-by`.
- Keep all edits outside `wideseek_r2` strictly additive (the `gigpo` registration/preprocessing/validation entries) and make no changes to `wideseek_r1`, other agents, or `rlinf/workers/agent/agent_loop.py`.

--- Original Design Draft Start ---

Here's the document translated into standard English markdown:

---

# Task: Create wideseek_r2 from wideseek_r1 and Refactor the Code

I want you to base this on `wideseek_r1`: first make a complete copy to generate `wideseek_r2`, then clean up the code by removing all the complex credit assignment logic.

The work is divided into two parts.

## Part 1: Copy wideseek_r1 and rename it to wideseek_r2

The following code involves `wideseek_r1`. Copy it and rename the copy to `wideseek_r2`:

1. `examples/agent/wideseek_r1`
2. `rlinf/agents/wideseek_r1`
3. `rlinf/data/datasets/wideseek_r1.py`
4. Copy `compute_grpo_dynamic_advantages` in `rlinf/algorithms/advantages.py`. The contents should remain identical for now, but name it `gigpo`. You also need to update the `adv_type` in the YAML config.

After copying, update all files, directories, class names, etc. to use the name `wideseek_r2`.

## Part 2: Refactor and optimize the code

Make the code clearer and easier to understand, and remove the complex credit-assignment-related code. Specifically:

1. **English-only prompts.** Our prompts should only consider English, not Chinese. Therefore:
   - In `rlinf/data/datasets/wideseek_r1.py`, delete everything related to `self.enable_zh`.
   - In `rlinf/agents/wideseek_r1/utils/prompt.py`, `rlinf/agents/wideseek_r1/utils/prompt_utils.py`, and `rlinf/agents/wideseek_r1/utils/tool_description.py`, delete all Chinese-related parts (everything `ZH`-related). The only language should be English!

2. **Make `add_few_shot` configurable.** In `rlinf/agents/wideseek_r1/utils/prompt_utils.py`, `add_few_shot` is currently set equal to `is_markdown`. I want you to add a variable in the config and in the upstream code so that it becomes an optional configuration.

3. **Don't hardcode tool description limits.** My config has `max_workers_per_planner: 10` and `max_toolcall_per_worker: 5`. These are optional, but in the current `rlinf/agents/wideseek_r1/utils/tool_description.py` the values are hardcoded as "ten" and "five". I want `tool_description` to stop hardcoding these and instead use something like `to a maximum of {max_toolcall_per_worker} tool instances per call` and `creating a maximum of {max_workers_per_planner} sub-agents`. Use the numeric values.

4. **Remove complex credit assignment.** Only use the final item F1 and format reward for the computation.
   - Delete the following from the config, along with the related code:
     ```yaml
     call_search_reward: 0.05
     length_limit: 3000
     max_length_limit: 5000
     length_penalty: 0.1
     over_context_penalty: 0.2
     ```
   - Substantially rewrite the `credit_assignment` in `rlinf/agents/wideseek_r1/utils/reward.py`. Delete `train_buffer` and `not_training`. `credit_assignment` should compute the reward using only `llm_reward` and `format_reward`; delete all other logic.
   - In `rlinf/workers/agent/agent_loop.py`, also delete:
     ```python
     if single_turn_output.extra_fields.get("not_training", False):
         continue
     ```
   - `credit_assignment` should return `reward_score`.
   - Then `final_answer_format` should use `answer_format = final_answer_extract is not None and format is True`.

5. **Simplify `run_one_query_role`.** The `run_one_query_role` function should no longer return `task_failed` or `succ_end`. Delete `_mark_role_failed_turns` as well. Delete `context_failed` and `tool_response_failed` too.

   **Note:** Add the following feature here. When a worker returns content to the main agent, it currently does this directly:
   ```python
   response_text.split("</think>")[-1].split("<|im_end|>")[0].strip()
   ```
   This isn't good. I require that the worker's return format must be:
   ```
   <answer>
   xxx
   </answer>
   ```
   To implement this, you need to state this clearly in the worker's system prompt, and add an extract function that extracts the content from the worker's final response text. The extraction can use `extract_final_answer`'s `mode == "tag"`. That way, if extraction succeeds, still use:
   ```python
   tool_messages_text.append(
       get_planner_subtask_result_message(...)
   )
   ```
   And if extraction fails, use:
   ```python
   tool_messages_text.append(
       get_planner_subtask_failed_message(...)
   )
   ```
   You need to implement this feature as well!

6. **Reorganize utility functions.** Create a new `utils.py` under `rlinf/agents/wideseek_r1/utils`:
   - Move `gen_extra_fields`, `_build_tool_call_info`, and `_set_max_turns` (with `self.cfg.agentloop` passed as a parameter) from `rlinf/agents/wideseek_r1/wideseek_r1.py`.
   - Move `_build_message_history_and_tools` into `rlinf/agents/wideseek_r1/utils/prompt_utils.py`.
   - Move `get_rollout_metrics` into `rlinf/agents/wideseek_r1/utils/metrics.py`.
   - Take lines 719–752 of `rlinf/agents/wideseek_r1/wideseek_r1.py`, write them as a standalone function, and put that function in `rlinf/agents/wideseek_r1/utils/utils.py`.

7. **Rename `is_markdown` to `answer_mode`.** There's a parameter `is_markdown` that runs through both the YAML and the code. Its original purpose: the current problems come in two types — one is a table, the other is a single element — and this was used to distinguish them. But this isn't extensible. I require that everywhere this appears, it be renamed to `answer_mode`, which currently supports two values: `markdown` and `boxed` (similar to the way `extract_final_answer` is written).

**Note:** All of the above modifications refer to the copied code under `wideseek_r2`. The `wideseek_r1`-related code must remain unchanged!

## Acceptance Criteria

The refactored code must support training and support eval with `mode=markdown/boxed`.

1. **Training script:** `bash examples/agent/wideseek_r2/run_train.sh`
   - Requirement: I should be able to observe one step training normally, with a training reward score of 0.3 or above.

2. **Eval scripts:**
   ```bash
   bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_width
   bash examples/agent/wideseek_r1/run_eval.sh eval_qwen3_qa
   ```
   - Requirement: Eval should finish normally with no problems in metric logging. The LLM reward should be 0.3 or above in all cases.
--- Original Design Draft End ---
