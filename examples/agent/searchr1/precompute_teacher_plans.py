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

"""Pre-generate frozen Search-R1 teacher plans on one GPU."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from rlinf.agents.searchr1.teacher_planner import (
    FrozenTeacherPlanner,
    load_teacher_questions,
    teacher_plan_cache_key,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", action="append", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--teacher-version", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--data-size", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-plan-chars", type=int, default=6144)
    parser.add_argument("--max-field-chars", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--require-plan",
        action="store_true",
        help="Reject KEEP outputs when every evaluation item is known to be multi-hop.",
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="Regenerate cache entries that exist but failed plan validation.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    return parser


def main() -> None:
    """Generate missing plans and atomically populate the shared cache."""
    args = build_parser().parse_args()
    if (
        args.batch_size <= 0
        or args.max_running_requests <= 0
        or args.max_steps < 2
        or args.max_attempts < 1
    ):
        raise ValueError(
            "batch sizes/max_attempts must be positive and max_steps at least 2"
        )

    questions = load_teacher_questions(args.data_path, args.prompt_key, args.data_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    cfg = OmegaConf.create(
        {
            "data": {"seed": args.seed},
            "teacher_planner": {
                "version": args.teacher_version,
                "seed": args.seed,
                "cache_dir": args.cache_dir,
                "cache_only": False,
                "max_new_tokens": args.max_new_tokens,
                "max_plan_chars": args.max_plan_chars,
                "max_field_chars": args.max_field_chars,
                "max_steps": args.max_steps,
                "max_attempts": args.max_attempts,
                "require_plan": args.require_plan,
            },
        }
    )
    planner = FrozenTeacherPlanner(cfg, tokenizer)

    pending_by_id: dict[str, str] = {}
    existing = 0
    for question in questions:
        plan_id = teacher_plan_cache_key(
            question, planner.teacher_version, planner.seed
        )
        cached = planner.cache.get(plan_id)
        if cached is not None and (cached.valid or not args.retry_invalid):
            existing += 1
        else:
            pending_by_id.setdefault(plan_id, question)

    print(
        json.dumps(
            {
                "questions": len(questions),
                "unique_missing": len(pending_by_id),
                "existing": existing,
                "cache_dir": str(Path(args.cache_dir).expanduser()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not pending_by_id:
        return

    # Importing SGLang starts CUDA-aware runtime initialization, so keep it out
    # of module import paths used by CPU-only unit tests.
    from rlinf.workers.rollout.sglang import Engine

    engine = Engine(
        model_path=args.model_path,
        tp_size=1,
        mem_fraction_static=args.gpu_memory_utilization,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        disable_cuda_graph=args.disable_cuda_graph,
        max_running_requests=args.max_running_requests,
        skip_tokenizer_init=False,
        log_level="info",
    )
    generated = 0
    generation_attempts = 0
    valid = 0
    invalid = 0
    pending_questions = list(pending_by_id.values())
    try:
        progress = tqdm(total=len(pending_questions), desc="Teacher plans")
        for start in range(0, len(pending_questions), args.batch_size):
            batch_questions = pending_questions[start : start + args.batch_size]
            pending = [
                (question, planner.build_prompt_ids(question))
                for question in batch_questions
            ]
            for attempt in range(args.max_attempts):
                outputs = engine.generate(
                    input_ids=[prompt_ids for _, prompt_ids in pending],
                    sampling_params=planner.sampling_params,
                )
                if isinstance(outputs, dict):
                    outputs = [outputs]
                if len(outputs) != len(pending):
                    raise RuntimeError(
                        "SGLang teacher output count does not match the input batch"
                    )
                generation_attempts += len(outputs)
                retry: list[tuple[str, list[int]]] = []
                for (question, _), output in zip(pending, outputs, strict=True):
                    raw_response = tokenizer.decode(
                        output["output_ids"], skip_special_tokens=True
                    )
                    result = planner.cache_response(question, raw_response)
                    if result.valid:
                        generated += 1
                        valid += 1
                        progress.update(1)
                    elif attempt + 1 < args.max_attempts:
                        retry.append(
                            (
                                question,
                                planner.build_repair_prompt_ids(
                                    question,
                                    raw_response,
                                    result.error or "unknown validation error",
                                ),
                            )
                        )
                    else:
                        generated += 1
                        invalid += 1
                        progress.update(1)
                pending = retry
                if not pending:
                    break
        progress.close()
    finally:
        engine.shutdown()

    print(
        json.dumps(
            {
                "generated": generated,
                "generation_attempts": generation_attempts,
                "valid": valid,
                "invalid": invalid,
                "valid_rate": valid / generated if generated else 0.0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
