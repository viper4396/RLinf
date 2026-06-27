#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Eval config name is passed as the first argument (e.g. qwen3-eval-markdown
# or qwen3-eval-boxed); defaults to the boxed eval.
CONFIG_NAME="${1:-qwen3-eval-boxed}"

python ${REPO_PATH}/examples/agent/wideseek_r2/eval.py --config-path ${REPO_PATH}/tests/e2e_tests/agent/wideseek_r2 --config-name ${CONFIG_NAME}
