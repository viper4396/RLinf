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