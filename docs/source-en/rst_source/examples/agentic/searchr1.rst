Reinforcement Learning Training of Search-R1
================================================

Multi-turn RL with tool calls has been proven to extend the interaction boundary of large language models (LLMs) to the real world.  
This document describes how to reproduce the experiments from  
`Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning <https://arxiv.org/abs/2503.09516>`__  
under the RLinf framework, using reinforcement learning (RL) to train LLMs to answer questions by invoking search tools.

Environment
-----------

RLinf Environment
~~~~~~~~~~~~~~~~~

RLinf environment setup follows the
:doc:`RLinf installation guide <../../start/installation>`.

Local Wiki Server Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the local retrieval server from the Search-R1 example.  
Install faiss via conda; details in  
`SearchR1 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__  
and installation reference in  
`Search-R1 & veRL-SGLang <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__  
The environment is also configured via conda.

.. code-block:: bash

   conda create -n retriever python=3.10 -y
   conda activate retriever

   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install transformers datasets pyserini huggingface_hub

   # Install GPU version of faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   pip install uvicorn fastapi

Wiki Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~

We use the local retrieval files provided by Asearcher.
The downloaded files are approximately 50–60 GB in size.

.. code-block:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/agent/searchr1/download.py --save_path $save_path

Download the `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding model from HuggingFace,  
and build the index

.. code-block:: bash

   bash examples/agent/tools/search_local_server_faiss/build_index.sh

Write the paths to the previously downloaded wiki files and the index into examples/agent/searchr1/launch_local_server.sh

.. code-block:: bash

   #!/bin/bash

   set -ex

   WIKI2018_WORK_DIR=$save_path

   index_file=$WIKI2018_WORK_DIR/e5.index/e5_Flat.index
   corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
   pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl
   retriever_name=e5
   retriever_path=path/to/intfloat/e5-base-v2

   python3  ./local_retrieval_server.py --index_path $index_file \
                                               --corpus_path $corpus_file \
                                               --pages_path $pages_file \
                                               --topk 3 \
                                               --retriever_name $retriever_name \
                                               --retriever_model $retriever_path \
                                               --faiss_gpu --port 8000

Run `launch_local_server.sh` to start the Local Wiki Server.  
Wait until server IP information is printed — indicating successful startup.

(Optional) Using Qdrant as Local Wiki Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also support qdrant as the wiki server as well. If you don't want to use the qdrant, move on to the Training section.

Download the local retrieval wiki corpus files provided by ASearcher using the method mentioned in the previous section.

Download the `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding model from HuggingFace.

Download `qdrant <https://github.com/qdrant/qdrant/releases>`__ binary file and build a qdrant collection with follwing steps. First, Create a new folder and put the qdrant binary into this folder, to facilitate the subsequent storage of qdrant binary and constructed collection files.

In `examples/agent/tools/search_local_server_qdrant/build_index_qdrant.sh` and `examples/agent/tools/search_local_server_qdrant/launch_local_server_qdrant.sh`, update the file paths for `WIKI2018_DIR`, `retriever_path`, and `qdrant_path` according to your downloaded wiki corpus, e5-base-v2, and qdrant paths.

Use the following commands to build the qdrant wiki server collection:

.. code-block:: bash

   # Create folder for qdrant
   mkdir -p /path/to/qdrant
   # Copy the binary
   cp qdrant /path/to/qdrant

   # Launch qdrant server
   /path/to/qdrant/qdrant &

   # Build qdrant collection
   bash examples/agent/tools/search_local_server_qdrant/build_index_qdrant.sh

Run launch_local_server_qdrant.sh to start the Local Qdrant Wiki Server. Wait until server IP information is printed — indicating successful startup.

.. code-block:: bash

   # Launch qdrant server
   /path/to/qdrant/qdrant &

   # Launch qdrant-based wiki server
   bash examples/agent/tools/search_local_server_qdrant/launch_local_server_qdrant.sh

Qdrant uses the HNSW graph index algorithm by default. For details on optimizing the HNSW graph index, please refer to the `Qdrant documentation <https://qdrant.tech/documentation/guides/optimize/>`__.


Training on 8×H100
------------------

Download the `training dataset <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__ from HuggingFace  
and write its path into `examples/agent/searchr1/config/train_qwen2.5.yaml`:

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/train.jsonl"]

Modify `rollout.model.model_path` in `train_qwen2.5.yaml`:

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
       model_type: qwen2.5

If you use `sampling_params.stop` to control model stop and save training time, detokenize should be set to True.

.. code-block:: yaml

   rollout:
      ……
      distributed_executor_backend: mp   # ray or mp
      disable_log_stats: False
      detokenize: True  

Since Search-R1 will re-tokenize the model output, ``recompute_logprobs`` should be set to True.

.. code-block:: yaml

   algorithm:
      ……
      recompute_logprobs: True
      shuffle_rollout: False

Reward and tool-response handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search-R1 computes normalized exact-match reward in a dedicated reward worker.
Ground-truth references travel through a reward-only channel and are not sent to
the agent loop or model prompt. The final reward is written only to the terminal
model turn of each trajectory.

``agentloop.max_tool_response_length`` is a token limit. With
``tool_response_truncate_side: right``, tokens on the right are removed and the
prefix is retained; ``left`` retains the suffix, and ``middle`` retains both
ends. ``max_turns: 2`` allows one search turn followed by one answer turn.

.. code-block:: yaml

   agentloop:
     max_turns: 2
     reward_mode: trajectory
     max_tool_response_length: 500
     tool_response_truncate_side: right

Run `bash examples/agent/searchr1/run_train.sh` to start training.

Evaluation
----------

Run the following commands to convert a Megatron checkpoint into a HuggingFace model:

.. code-block:: bash

   CKPT_PATH_MG={your_output_dir}/{exp_name}/checkpoints/global_step_xxx/actor
   CKPT_PATH_HF={path/to/save/huggingface/model}
   CKPT_PATH_ORIGINAL_HF={path/to/model/Qwen2.5-3B-Instruct}
   CKPT_PATH_MF="${CKPT_PATH_HF}_middle_file"

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_mg_to_middle_file \
       --load-path "${CKPT_PATH_MG}" \
       --save-path "${CKPT_PATH_MF}" \
       --model qwen_2.5_3b \
       --tp-size 1 --ep-size 1 --pp-size 1 \
       --te-ln-linear-qkv true --te-ln-linear-mlp_fc1 true \
       --te-extra-state-check-none true --use-gpu-num 0 --process-num 16

   python -m rlinf.utils.ckpt_convertor.megatron_convertor.convert_middle_file_to_hf \
       --load-path "${CKPT_PATH_MF}" \
       --save-path "${CKPT_PATH_HF}" \
       --model qwen_2.5_3b \
       --use-gpu-num 0 --process-num 16

   rm -rf "${CKPT_PATH_MF}"
   rm -f "${CKPT_PATH_HF}"/*.done
   shopt -s extglob
   cp "${CKPT_PATH_ORIGINAL_HF}"/!(*model.safetensors.index.json) "${CKPT_PATH_HF}"

Fill the converted HuggingFace model path into  
`examples/agent/searchr1/config/eval_qwen2.5.yaml`:

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/eval/model
       model_type: qwen2.5

Modify the evaluation dataset path:

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/eval.jsonl"]
     val_data_paths: ["/path/to/eval.jsonl"]

Run `bash examples/agent/searchr1/run_eval.sh` to start evaluation.

BrowseComp with Serper, Jina, and an external judge
----------------------------------------------------

``eval_browsecomp_online.yaml`` evaluates a Search-R1 checkpoint on raw
BrowseComp JSONL records with ``question`` and ``answer`` fields. The
``searchr1`` dataset adapter applies the original ``local-rag`` ChatML prompt.
The search worker queries Serper, enriches the highest-ranked pages through
Jina, and filters result URLs matching the configured benchmark-mirror
patterns. The main accuracy uses the external OpenAI-compatible judge, while
``eval/exact_match`` retains the normalized Search-R1 exact-match score.

Export the API keys and proxy variables before starting Ray, because Ray
workers inherit the environment captured at startup:

.. code-block:: bash

   export SERPER_API_KEY=...
   export JINA_API_KEY=...
   export http_proxy=...
   export https_proxy=...
   export RLINF_NODE_RANK=0
   ray start --head

The example config uses
``/mnt/public/suheng/data/browsecomp_padded1280.jsonl``,
``/mnt/public/suheng/model/search_r1``, and the judge at
``http://172.27.22.136:30000``. Run an eight-sample smoke test first:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_browsecomp_online \
     data.data_size=8 data.val_rollout_batch_size=8 \
     runner.experiment_name=searchr1-browsecomp-online-smoke8

After the smoke test succeeds, run all 1280 records:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_browsecomp_online

Results are saved to
``/mnt/public/suheng/searchr1_runs/<experiment_name>/eval_results.json``.
The top-level ``accuracy`` and ``eval/judge_accuracy`` use judge equivalence;
``eval/exact_match`` reports normalized exact match. Per-trajectory output
retains both scores and the judge response.

GISA structured-answer evaluation
---------------------------------

``eval_gisa_online.yaml`` evaluates the Search-R1 checkpoint on
``/mnt/public/suheng/data/gisa_full373.jsonl`` with the same Serper and Jina
online tools. GISA records preserve their per-example ``answer_type``
(``table``, ``set``, ``list``, or ``item``), ``is_markdown``, and
``unique_columns`` metadata. The final answer must remain inside
``<answer>...</answer>``; structured answers use one fenced Markdown table.

GISA uses the same OpenAI-compatible judge endpoint as BrowseComp, but applies
the semantic structured-answer scorer shared with WideSeek-R2. It judges
items, cells, row keys, and complete rows before aggregation, and reports
``gisa/cell_f1``, ``gisa/exact_match``, ``gisa/pass@1``,
``gisa/format_rate``, ``gisa/table_row_f1``, and
``gisa/list_order_score``, plus per-answer-type metrics. The example points to
``http://172.27.22.136:30000`` and
``Qwen3-30B-A3B-Instruct-2507``.

Run an eight-sample smoke test first:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_gisa_online \
     data.data_size=8 data.val_rollout_batch_size=8 \
     runner.experiment_name=searchr1-gisa-online-smoke8

After the smoke test succeeds, evaluate all 373 records:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_gisa_online

``agentloop.context_safety_margin`` keeps each prompt-plus-completion request
strictly below the model context limit. After every completed batch, the runner
atomically updates ``eval_results.partial.json`` in the experiment directory so
completed trajectories remain available if a later batch fails.

Results are saved to
``/mnt/public/suheng/searchr1_runs/<experiment_name>/eval_results.json``.

Frozen Teacher-Planner Shadow A/B
---------------------------------

The phase-2 shadow evaluation keeps the policy frozen and launches an
independent ``teacher_planner`` rollout using Qwen2.5-7B-Instruct. The teacher
receives the question but never the ground-truth answer. Its strict JSON output
contains ``decision``, ``plan_type``, and an ordered ``steps`` list. Every step
records ``step_id``, ``goal``, ``query_template``, ``expected_evidence``, and
``depends_on``. A dependent query uses placeholders such as
``{step_1_result}``, which the policy must instantiate from actual retrieval
evidence instead of a teacher guess. Plans are cached by question, teacher
version, and seed. ``KEEP`` represents a direct single-hop question and leaves
the policy prompt unchanged. ``PLAN`` requires two to eight validated sequential
or comparison hops. The planner retries rejected outputs with a question-only
repair prompt up to ``teacher_planner.max_attempts``. Set ``require_plan: true``
for an all-multihop benchmark such as 2WikiMultihopQA so an accidental ``KEEP``
is retried as well. A plan that remains invalid safely falls back to no guidance.

The legacy ``execution_mode: prompt`` inserts the complete plan as a separate
low-trust ChatML user message before restoring the assistant-generation prefix.
Controller mode uses the same validated ChatML boundary for current-hop and
synthesis messages but never exposes the complete plan to the policy.

With ``execution_mode: controller``, an accepted ``PLAN`` becomes an enforced
state machine instead of persistent prompt text. The controller executes every
independent root hop exactly from its validated query template. For a dependent
hop, it exposes only that hop and the evidence from its declared dependencies;
the controller requests a strict JSON binding with ``resolved_values`` and a
fully bound ``query``. Each ``step_N_result`` value must be grounded in that
specific dependency's evidence, and the query must contain the value or a
recorded normalized alias. Invalid bindings are retried up to
``controller_bind_max_attempts``. A premature answer, malformed JSON,
cross-dependency value, or unresolved placeholder cannot terminate the plan.
The recorded fallback is candidate-specific: it combines the current
goal with dependency-step goals, templates, expected evidence, and retrieved
document titles instead of repeating the complete original question. After
every hop completes, RLinf removes the plan and builds a separate synthesis
prompt from the original question and bounded collected evidence. The
controller places the opening ``<think>`` in the generation prompt, requires a
short evidence-grounded derivation followed by one ``<answer>`` tag, normalizes
the final tag, and makes only that isolated synthesis response visible to reward
evaluation. Rejected intermediate answer tags can therefore never become the
final answer. Comparison synthesis
explicitly asks for the requested candidate or yes/no decision instead of an
intermediate date, number, or name.

``agentloop.max_turns`` must be at least ``teacher_planner.max_steps + 1`` so
each configured hop still leaves one final synthesis turn. The example uses
``max_turns: 10`` and raises ``runner.seq_length`` to 12288. Root-hop controller
outputs are metadata-only and excluded from policy training tensors. The result
JSON records controller phase, query source (``template``, ``policy``, or
``fallback``), completed step IDs, resolved values, binding attempts and stable
failure reasons, synthesis format repair, and whether synthesis completed.

Set the policy and teacher model paths in
``examples/agent/searchr1/config/eval_teacher_shadow_qwen2.5.yaml``. The default
layout uses hardware ranks 0--7 for policy evaluation and rank 8 for the frozen
teacher, so it requires two Ray nodes when each node has eight GPUs. Adjust
``cluster.component_placement`` when using pre-generated plans or a different
teacher tensor-parallel size.

The main paired run keeps four trajectories per question:

.. code-block:: yaml

   algorithm:
     group_size: 4

   agentloop:
     max_turns: 10
     force_search_on_first_turn: true

   teacher_planner:
     enabled: true
     execution_mode: controller
     max_attempts: 3
     require_plan: true
     controller_max_evidence_length: 6000
     controller_min_synthesis_tokens: 256
     controller_bind_max_attempts: 3
     dual_query_retrieval: false
     use_fallback_query: false
     guidance_modes: [guided, guided, unguided, unguided]

Run the paired evaluation with:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5

On a single eight-GPU node, generate the plans on one GPU first, release that
GPU, and then run the policy on all eight GPUs. The cache-only configuration
fails immediately if a plan is missing instead of launching a teacher model:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python examples/agent/searchr1/precompute_teacher_plans.py \
     --model-path /path/to/Qwen2.5-7B-Instruct \
     --data-path /path/to/eval.jsonl \
     --cache-dir ../results/teacher_plan_cache/qwen2.5-7b-instruct-multihop-v5-seed1234 \
     --teacher-version qwen2.5-7b-instruct-multihop-v5 --seed 1234 \
     --max-attempts 3 --require-plan --retry-invalid

   bash examples/agent/searchr1/run_eval.sh \
     eval_teacher_shadow_qwen2.5_8gpu \
     rollout.model.model_path=/path/to/policy \
     teacher_planner.model.model_path=/path/to/Qwen2.5-7B-Instruct \
     data.val_data_paths='[/path/to/eval.jsonl]'

Run the two placebo controls separately by overriding ``guidance_modes``:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5 \
     teacher_planner.guidance_modes='[shuffled,shuffled,unguided,unguided]'
   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5 \
     teacher_planner.guidance_modes='[generic,generic,unguided,unguided]'

With controller execution, ``shuffled`` runs an unrelated validated plan through
the same state machine. ``generic`` remains a length-matched prompt-perturbation
control and does not execute the semantic plan.

The result summary contains ``planner/guided_EM``,
``planner/unguided_EM``, ``planner/guided_minus_unguided``,
``planner/plan_valid_rate``, ``planner/query_change_rate``, and
``planner/answer_hit_delta``. Shuffled and generic runs emit the corresponding
mode-specific metrics. The summary also reports deterministic paired-bootstrap
95% CI bounds as ``planner/<mode>_uplift_ci_low`` and
``planner/<mode>_uplift_ci_high``. Use at least two questions per agent-loop
request for the shuffled control. ``planner/rewrite_rate``,
``search/<mode>_dual_query_rate``, and
``search/<mode>_tool_call_repair_rate`` expose the new gate and retrieval path.
``planner/<mode>_diagnostic_SubEM`` is a secondary diagnostic for answer-list
and alias artifacts; it does not change the primary exact-match reward.
``planner/<mode>_controller_completion_rate`` reports whether all hops reached
synthesis, while ``search/<mode>_controller_fallback_query_rate`` reports the
fraction of all completed hops that required controller repair.
``search/<mode>_controller_dependent_fallback_rate`` uses only dependent hops as
the denominator. ``search/<mode>_dependent_query_binding_valid_rate``,
``search/<mode>_binding_attempts_per_dependent_hop``, binding failure-reason
rates, and ``search/<mode>_unresolved_placeholder_rate`` expose the dependent
binding protocol directly. ``planner/<mode>_synthesis_format_valid_rate`` and
``planner/<mode>_synthesis_format_repair_rate`` report final answer protocol
validity and deterministic answer-tag normalization. Label-only post-processing
adds ``planner/plan_semantic_coverage_rate`` and per-type EM, answer-hit, gold
evidence-object coverage, paired uplift, and confidence intervals under
``planner/type/<question_type>/...`` and ``search/type/<question_type>/...``.
The dataset labels are joined only after rollout and never enter teacher,
retrieval, or synthesis inputs.

Before saving, the runner rejects missing questions, duplicate trajectories, or
unbalanced A/B arms. The result summary stores dataset, plan-cache, controller,
resolved-config, model-manifest, and retrieval-config hashes, plus search/turn/
generated-token budgets and wall times. ``acceptance/A_pass``,
``acceptance/B_pass``, ``acceptance/C_pass``, and ``acceptance/ABC_pass`` apply
the frozen phase-2 thresholds to these metrics. Teacher shadow configs must
therefore provide ``tools.search.index_manifest.manifest_sha256``.

Training Curves
---------------

The following shows the reward curves and training time curves.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/searchr1.png" style="width: 100%;"/>
       <p><em>Qwen2.5-3B-Instruct in RLinf</em></p>
     </div>
   </div>

Compared to the original performance (133s per step after response length stabilizes), we achieved a 55% speedup while maintaining consistent reward curves and evaluation results.

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/searchr1_orig_impl_time.png" style="width: 35%;"/>
       <p><em>Qwen2.5-3B-Instruct in original implementation at PeterGriffinJin/Search-R1</em></p>
     </div>
   </div>

References
----------

search-r1: https://github.com/PeterGriffinJin/Search-R1

Search-R1 & veRL-SGLang:  
https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md

Asearcher: https://github.com/inclusionAI/ASearcher
