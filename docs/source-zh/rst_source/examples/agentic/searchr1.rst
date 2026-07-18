Search-R1的强化学习训练
=======================

结合工具调用的Multi-turn
RL被证明能够将大语言模型（LLM）的交互边界扩展到真实世界。本文档介绍了如何在
RLinf 框架下复现论文\ `Search-R1: Training LLMs to Reason and Leverage
Search Engines with Reinforcement
Learning <https://arxiv.org/abs/2503.09516>`__\ 中的实验，使用强化学习（RL）来训练大语言模型（LLM）通过调用搜索工具回答问题。

环境
----

RLinf环境
~~~~~~~~~

RLinf 环境配置参照 :doc:`RLinf 安装指南 <../../start/installation>`。

Local Wiki Server运行环境
~~~~~~~~~~~~~~~~~~~~~~~~~

我们使用search-R1示例中的local retrieve
server，通过conda安装faiss，详细文档见\ `SearchR1 <https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md>`__\ ，安装过程参考\ `Search-R1 &
veRL-SGLang <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md>`__\ ，同样使用conda来配置环境

.. code-block:: bash

   conda create -n retriever python=3.10 -y
   conda activate retriever

   conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install transformers datasets pyserini huggingface_hub

   #  安装 GPU 版 faiss
   conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

   pip install uvicorn fastapi

Wiki配置文件
~~~~~~~~~~~~

我们使用Asearcher提供的本地检索文件，下载文件大约 50~60GB

.. code-block:: bash

   conda activate retriever

   save_path=/the/path/to/save
   python examples/agent/searchr1/download.py --save_path $save_path

从huggingface上下载\ `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding模型，并生成index

.. code-block:: bash

   bash examples/agent/tools/search_local_server_faiss/build_index.sh

将之前下载好的wiki文件路径和index路径等写入examples/agent/searchr1/launch_local_server.sh

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

运行launch_local_server.sh启动Local Wiki Server，等待直至输出server ip等信息，代表server启动完成

(Optional) 使用Qdrant作为Wiki Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们也支持使用 qdrant 作为 wiki 服务器。如果你不打算使用 qdrant，可以直接跳到 训练 部分。

使用上一部分中提到的方式准备 Asearcher 提供的本地 wiki corpus 检索文件。

从 huggingface 上下载\ `e5-base-v2 <https://huggingface.co/intfloat/e5-base-v2>`__ embedding 模型。

下载 `qdrant <https://github.com/qdrant/qdrant/releases>`__ 并按照以下步骤构建 qdrant collection。首先，创建一个文件夹并把下载好的 qdrant 二进制文件放入该文件夹中，方便后续存储 qdrant 程序及其构建的 collection 文件。

在 `examples/agent/tools/search_local_server_qdrant/build_index_qdrant.sh` 和 `examples/agent/tools/search_local_server_qdrant/launch_local_server_qdrant.sh` 中，根据之前下载的 wiki corpus, e5-base-v2 和 qdrant 路径更新 `WIKI2018_DIR`、 `retriever_path` 和 `qdrant_path` 的文件路径。

使用以下指令构建 qdrant wiki 服务器的 collection：

.. code-block:: bash

   # 创建 qdrant 存放的文件夹
   mkdir -p /path/to/qdrant
   # 拷贝二进制执行文件
   cp qdrant /path/to/qdrant

   # 启动 qdrant server
   /path/to/qdrant/qdrant &

   # 构建 qdrant collection
   bash examples/agent/tools/search_local_server_qdrant/build_index_qdrant.sh

运行 launch_local_server_qdrant.sh 启动 Local Qdrant Wiki Server ，等待直至输出 server ip 等信息，代表 server 启动完成

.. code-block:: bash

   # 启动 qdrant server
   /path/to/qdrant/qdrant &

   # 启动基于 qdrant 的 wiki server
   bash examples/agent/tools/search_local_server_qdrant/launch_local_server_qdrant.sh

Qdrant 默认使用 HNSW 图索引算法。关于 HNSW 图索引的优化,请参考 `Qdrant 文档 <https://qdrant.tech/documentation/guides/optimize/>`__。

在8*H100上训练
--------------

从huggingface上下载\ `训练集 <https://huggingface.co/datasets/RLinf/Search-R1-Data>`__
，并将路径写入 `examples/agent/searchr1/config/train_qwen2.5.yaml`:

.. code-block:: yaml

   data:
     ……
     train_data_paths: ["/path/to/train.jsonl"]

修改 `train_qwen2.5.yaml` 中 `rollout.model.model_path` 的路径

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/model/Qwen2.5-3B-Instruct
       model_type: qwen2.5

如果使用 `sampling_params.stop` 来控制模型停止节省训练时间，detokenize应当设置为True

.. code-block:: yaml

   rollout:
      ……
      distributed_executor_backend: mp   # ray or mp
      disable_log_stats: False
      detokenize: True  

由于 Search-R1 会re-tokenize模型输出， `recompute_logprobs` 应当设置为True

.. code-block:: yaml

   algorithm:
      ……
      recompute_logprobs: True
      shuffle_rollout: False

Reward 与工具响应处理
~~~~~~~~~~~~~~~~~~~~~

Search-R1 在独立的 reward worker 中计算归一化精确匹配奖励。GT reference
仅通过 reward 专用 channel 传输，不会发送给 agent loop 或模型 prompt。每条轨迹的
最终奖励只写入 terminal model turn。

``agentloop.max_tool_response_length`` 表示 token 数上限。当
``tool_response_truncate_side: right`` 时，删除右侧 token 并保留前缀；``left``
保留后缀，``middle`` 保留两端。``max_turns: 2`` 允许一次搜索 turn 后接一次回答
turn。

.. code-block:: yaml

   agentloop:
     max_turns: 2
     reward_mode: trajectory
     max_tool_response_length: 500
     tool_response_truncate_side: right

运行 `bash examples/agent/searchr1/run_train.sh` 启动训练。

测试
----

运行以下命令将 Megatron checkpoint 转换为 HuggingFace model

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

将转换得到的huggingface
model路径填入 `examples/agent/searchr1/config/eval_qwen2.5.yaml`

.. code-block:: yaml

   rollout:
     group_name: "RolloutGroup"

     gpu_memory_utilization: 0.8
     model:
       model_path: /path/to/eval/model
       model_type: qwen2.5

修改测试数据集路径

.. code-block:: yaml

   data:
     ……
     val_data_paths: ["/path/to/eval.jsonl"]

运行 `bash examples/agent/searchr1/run_eval.sh` 启动测试。

冻结 Teacher Planner 的 Shadow A/B
----------------------------------

阶段 2 的 shadow 评估保持 policy 冻结，并使用 Qwen2.5-7B-Instruct 启动独立的
``teacher_planner`` rollout。Teacher 只接收问题，不会接收 GT。严格 JSON 输出包含
``decision``、``plan_type`` 和有序 ``steps``。每一步记录 ``step_id``、``goal``、
``query_template``、``expected_evidence`` 和 ``depends_on``。依赖前序结果的查询使用
``{step_1_result}`` 形式的占位符，policy 必须根据真实检索证据替换，不能使用 teacher
猜测的中间实体。Plan 按问题、teacher 版本和 seed 缓存。``KEEP`` 表示直接 single-hop
问题，不改变 policy prompt；``PLAN`` 必须包含 2 到 8 个通过校验的 sequential 或
comparison hop。Planner 会使用只含问题的 repair prompt 重试被拒绝的输出，最多执行
``teacher_planner.max_attempts`` 次。对于 2WikiMultihopQA 这类全 multihop benchmark，
设置 ``require_plan: true`` 后，意外生成的 ``KEEP`` 也会被重试。最终仍无效的 plan
会安全退化为不加 guidance。

旧的 ``execution_mode: prompt`` 会将完整 plan 作为独立的低权限 ChatML user message
插入，再恢复 assistant generation prefix。Controller mode 会使用相同的边界校验来
插入 current-hop 和 synthesis message，但不会向 policy 暴露完整 plan。

设置 ``execution_mode: controller`` 后，通过门控的 ``PLAN`` 会成为强制执行的状态机，
而不是跨 turn 持续保留的 prompt 文本。Controller 会直接按照通过校验的 query template
执行每个无依赖 root hop。对于 dependent hop，只向 policy 展示当前 hop 及其声明依赖的
证据；controller 要求 policy 返回包含 ``resolved_values`` 和完整 ``query`` 的严格 JSON
binding。每个 ``step_N_result`` 必须由对应 dependency 的证据支持，query 必须包含该值或
有记录的规范化 alias。无效 binding 最多重试 ``controller_bind_max_attempts`` 次。提前
回答、错误 JSON、跨 dependency 取值或未替换占位符都不能结束计划。可追踪的
fallback 不再重复完整原问题，而是组合当前 goal、依赖 step 的 goal、template、预期证据
和检索文档标题，形成候选特定的查询。所有 hop 完成后，RLinf 会移除 plan，使用原问题和
有长度上限的汇总证据构建独立 synthesis prompt。Controller 会将开头的 ``<think>``
放入 generation prompt，要求先输出简短且有证据支撑的推导，再输出唯一 ``<answer>``，
随后规范化最终 answer tag，并且只允许隔离后的 synthesis
response 进入 reward 评估；被拒绝的中间 answer tag 不会再成为最终答案。Comparison
synthesis 会明确要求返回候选实体或 yes/no，而不是日期、数字、导演名等中间属性。

``agentloop.max_turns`` 必须不小于 ``teacher_planner.max_steps + 1``，确保所有 hop 后仍有
一次最终 synthesis。示例使用 ``max_turns: 10``，并将 ``runner.seq_length`` 提高到
12288。Controller 生成的 root-hop 输出只作为元数据，不会进入 policy 训练 tensor。
结果 JSON 会记录 controller phase、query source（``template``、``policy`` 或
``fallback``）、已完成 step ID、resolved value、binding 尝试次数和稳定失败原因、
synthesis format repair，以及 synthesis 是否完成。

在 ``examples/agent/searchr1/config/eval_teacher_shadow_qwen2.5.yaml`` 中设置
policy 和 teacher 模型路径。默认布局使用硬件 rank 0--7 运行 policy 评估，rank 8
运行冻结 teacher；如果每个节点有 8 张 GPU，则需要两个 Ray 节点。使用预生成 plan
或不同 teacher tensor parallel size 时，应相应调整
``cluster.component_placement``。

主 paired 实验为每个问题保留四条轨迹：

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

运行 paired 评估：

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5

在单台 8 GPU 节点上，先用 1 张 GPU 生成 plan 并释放该 GPU，再使用
8 张 GPU 运行 policy。Cache-only 配置遇到 plan 缺失时会立即报错，
不会隐式启动 teacher 模型：

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

通过覆盖 ``guidance_modes`` 分别运行两个 placebo 对照：

.. code-block:: bash

   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5 \
     teacher_planner.guidance_modes='[shuffled,shuffled,unguided,unguided]'
   bash examples/agent/searchr1/run_eval.sh eval_teacher_shadow_qwen2.5 \
     teacher_planner.guidance_modes='[generic,generic,unguided,unguided]'

在 controller execution 下，``shuffled`` 会通过相同状态机执行一条无关但通过校验的
plan；``generic`` 仍是长度匹配的 prompt 扰动对照，不会执行语义 plan。

结果摘要包含 ``planner/guided_EM``、``planner/unguided_EM``、
``planner/guided_minus_unguided``、``planner/plan_valid_rate``、
``planner/query_change_rate`` 和 ``planner/answer_hit_delta``。Shuffled 和 generic
实验会输出相应的 mode-specific 指标。摘要还会通过
``planner/<mode>_uplift_ci_low`` 和 ``planner/<mode>_uplift_ci_high`` 输出可复现的
paired-bootstrap 95% CI。Shuffled 对照要求每个 agent-loop request 至少包含两个问题。
``planner/rewrite_rate``、``search/<mode>_dual_query_rate`` 和
``search/<mode>_tool_call_repair_rate`` 用于观测新门控和检索路径。
``planner/<mode>_diagnostic_SubEM`` 仅用于诊断答案列表和 alias 类评测误差，
不会改变主 exact-match reward。``planner/<mode>_controller_completion_rate``
表示所有 hop 是否都进入 synthesis，
``search/<mode>_controller_fallback_query_rate`` 表示已完成 hop 中需要 controller
修复查询的比例。``search/<mode>_controller_dependent_fallback_rate`` 只使用 dependent
hop 作为分母。``search/<mode>_dependent_query_binding_valid_rate``、
``search/<mode>_binding_attempts_per_dependent_hop``、binding failure-reason rate 和
``search/<mode>_unresolved_placeholder_rate`` 会直接呈现 dependent binding 协议质量。
``planner/<mode>_synthesis_format_valid_rate`` 和
``planner/<mode>_synthesis_format_repair_rate`` 分别表示最终答案协议有效率和确定性
answer-tag 规范化比例。Label-only 后处理还会增加
``planner/plan_semantic_coverage_rate``，并在
``planner/type/<question_type>/...`` 和 ``search/type/<question_type>/...`` 下输出各题型
EM、answer-hit、gold evidence-object coverage、paired uplift 和置信区间。数据集标签只在
rollout 完成后 join，不会进入 teacher、retrieval 或 synthesis 输入。

保存结果前，runner 会拒绝缺失问题、重复 trajectory 或不平衡的 A/B 臂。结果摘要会保存
dataset、plan cache、controller、resolved config、model manifest 和 retrieval config
的 hash，同时报告搜索次数、turn 数、生成 token 预算和 wall time。
``acceptance/A_pass``、``acceptance/B_pass``、``acceptance/C_pass`` 和
``acceptance/ABC_pass`` 会按冻结的阶段 2 阈值自动判定。因此 teacher shadow 配置必须提供
``tools.search.index_manifest.manifest_sha256``。

训练曲线
--------

下面展示 reward 曲线和训练时间曲线。

.. raw:: html

   <div style="display: flex; justify-content: space-between; gap: 10px;">
     <div style="flex: 1; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/searchr1.png" style="width: 100%;"/>
       <p><em>Qwen2.5-3B-Instruct in RLinf</em></p>
     </div>
   </div>

相较于原版性能( response length 稳定后，单 step 133s)，我们加速了 55%，同时 reward 曲线和 eval 结果保持一致。

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

Search-R1 &
veRL-SGLang:
https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md

Asearcher: https://github.com/inclusionAI/ASearcher
