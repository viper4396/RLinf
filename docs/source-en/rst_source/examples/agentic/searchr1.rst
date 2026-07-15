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

Frozen Teacher-Planner Shadow A/B
---------------------------------

The phase-2 shadow evaluation keeps the policy frozen and launches an
independent ``teacher_planner`` rollout using Qwen2.5-7B-Instruct. The teacher
receives the question but never the ground-truth answer. Its strict four-field
JSON plan is cached by question, teacher version, and seed, and is injected as
low-trust guidance only for the policy's first search.

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

   teacher_planner:
     enabled: true
     guidance_modes: [guided, guided, unguided, unguided]

Run the paired evaluation with:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5

Run the two placebo controls separately by overriding ``guidance_modes``:

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5 \
     teacher_planner.guidance_modes='[shuffled,shuffled,unguided,unguided]'
   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5 \
     teacher_planner.guidance_modes='[generic,generic,unguided,unguided]'

The result summary contains ``planner/guided_EM``,
``planner/unguided_EM``, ``planner/guided_minus_unguided``,
``planner/plan_valid_rate``, ``planner/query_change_rate``, and
``planner/answer_hit_delta``. Shuffled and generic runs emit the corresponding
mode-specific metrics. The summary also reports deterministic paired-bootstrap
95% CI bounds as ``planner/<mode>_uplift_ci_low`` and
``planner/<mode>_uplift_ci_high``. Use at least two questions per agent-loop
request for the shuffled control.

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
