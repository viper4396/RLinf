AZR 阶段 0：基线复现契约
========================

Absolute Zero Reasoner（AZR）迁移从冻结的复现契约开始。契约固定论文版
``origin/paper@41ed983`` 和 Qwen2.5-Coder-3B，独立于当前 AZR ``master``
默认配置以及 RLinf 通用 reasoning 配置。

冻结的训练语义
--------------

契约固定以下行为：

* FSDP Actor 与 vLLM Rollout，Tensor Parallel size 为 2；
* train batch size 为 64，prompt length 为 6144，response length 为 8096；
* PROPOSE rollout ``n=1``，难度评估 SOLVE rollout ``n=8``；
* Reinforce++ advantage、PPO clipped Actor loss，且不使用 Critic；
* ``code_i``、``code_o``、``code_f`` 三类任务；
* 难度评估轨迹只用于打分，不进入训练 batch。

已提交的展开配置位于
``examples/reasoning/config/absolute_zero/baseline_config.yaml``。

生成并校验基线
--------------

在 RLinf 仓库根目录执行：

.. code-block:: bash

   python scripts/azr/freeze_baseline.py \
     --azr-root ~/AZR \
     --rlinf-root /path/to/clean/rlinf-a099ff4a-worktree \
     --output-dir examples/reasoning/config/absolute_zero

命令会记录当前可用两个 seed 数据集的 SHA256。缺失的验证集或模型文件会
被明确标记为阻塞项，不会静默替换。准备好外部文件后执行：

.. code-block:: bash

   python scripts/azr/freeze_baseline.py \
     --azr-root ~/AZR \
     --rlinf-root /path/to/clean/rlinf-a099ff4a-worktree \
     --validation-data /path/to/test_answer.parquet \
     --model-checkpoint /path/to/Qwen2.5-Coder-3B \
     --baseline-trace /path/to/baseline_trace.jsonl \
     --output-dir examples/reasoning/config/absolute_zero

GPU 短跑前使用严格门禁：

.. code-block:: bash

   python scripts/azr/freeze_baseline.py \
     --validate-only examples/reasoning/config/absolute_zero/baseline_manifest.yaml \
     --strict

严格门禁还要求记录 20--100 步原 AZR 短跑。每一步必须保留 proposal
response、难度评估和正式 solve response、reward、有效题目以及 Program Pool
大小。缺少外部文件或短跑 trace 时，manifest 会保持 pending，这是有意的。
校验器还会检查每个 proposal 是否恰好关联 8 条难度评估 response。

产物
----

* ``baseline_manifest.yaml``：版本、哈希、环境、配置和门禁状态；
* ``fixed_examples.jsonl``：50 个确定性样例，包括 20 个 ``code_i``、20 个
  ``code_o`` 和 10 个 ``code_f``；
* ``baseline_config.yaml``：论文版 ``coder3b.sh`` 参数的显式展开。
