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

"""Tests for the Phase 0 AZR reproduction contract."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.azr.freeze_baseline import (
    DEFAULT_FIXED_COUNTS,
    MODEL_ID,
    PROBLEM_TYPES,
    build_fixed_examples,
    sha256_file,
    validate_manifest,
    validate_trace_records,
    write_jsonl,
)

_CONFIG_PATH = Path(__file__).parents[2] / (
    "examples/reasoning/config/absolute_zero/baseline_config.yaml"
)
_MANIFEST_PATH = Path(__file__).parents[2] / (
    "examples/reasoning/config/absolute_zero/baseline_manifest.yaml"
)
_FIXED_EXAMPLES_PATH = Path(__file__).parents[2] / (
    "examples/reasoning/config/absolute_zero/fixed_examples.jsonl"
)


def test_fixed_examples_are_deterministic_and_cover_all_problem_types():
    io_records = [
        {
            "snippet": f"def f(x): return {index}",
            "input": str(index),
            "output": str(index),
        }
        for index in range(3)
    ]
    code_f_records = [
        {"snippet": f"def f(x): return {index}", "inputs": [], "outputs": []}
        for index in range(2)
    ]

    first = build_fixed_examples(
        io_records,
        code_f_records,
        counts={"code_i": 2, "code_o": 2, "code_f": 1},
    )
    second = build_fixed_examples(
        io_records,
        code_f_records,
        counts={"code_i": 2, "code_o": 2, "code_f": 1},
    )

    assert first == second
    assert [item["problem_type"] for item in first] == [
        "code_i",
        "code_i",
        "code_o",
        "code_o",
        "code_f",
    ]


def test_generated_fixed_examples_have_the_frozen_coverage():
    with _FIXED_EXAMPLES_PATH.open(encoding="utf-8") as file:
        examples = [json.loads(line) for line in file if line.strip()]

    observed = {
        problem_type: sum(item["problem_type"] == problem_type for item in examples)
        for problem_type in PROBLEM_TYPES
    }
    assert observed == DEFAULT_FIXED_COUNTS
    assert len(examples) >= 50


def test_baseline_manifest_freezes_versions_and_configuration():
    with _CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with _MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = yaml.safe_load(file)

    assert manifest["repositories"]["azr"]["expected_commit"].startswith("41ed983")
    assert manifest["repositories"]["rlinf"]["expected_commit"].startswith("a099ff4")
    assert manifest["model"]["id"] == MODEL_ID
    assert manifest["configuration"]["resolved"] == config
    assert manifest["artifacts"]["seed_error"]["required"] is False
    assert config["data"]["train_batch_size"] == 64
    assert config["data"]["max_prompt_length"] == 6144
    assert config["data"]["max_response_length"] == 8096
    assert config["rollout"]["n"] == 1
    assert config["reward"]["difficulty_samples"] == 8
    assert config["algorithm"]["advantage"] == "reinforce_plus_plus"
    assert config["algorithm"]["critic"] is False
    assert config["rollout"]["tensor_parallel_size"] == 2
    assert manifest["fixed_examples"]["observed_counts"] == DEFAULT_FIXED_COUNTS
    assert (
        "difficulty_solve_responses"
        in manifest["short_baseline"]["required_per_step_fields"]
    )


def test_manifest_validator_reports_unfinished_short_run():
    with _MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = yaml.safe_load(file)

    errors = validate_manifest(manifest, strict=True)

    assert any("short run" in error for error in errors)
    assert any("missing artifact" in error for error in errors)


def test_trace_validator_preserves_proposal_to_difficulty_grouping():
    records = [
        {
            "global_step": step,
            "proposal_responses": [{"proposal_id": f"proposal-{step}"}],
            "difficulty_solve_responses": [
                {
                    "proposal_id": f"proposal-{step}",
                    "responses": [f"solve-{index}" for index in range(8)],
                }
            ],
            "formal_solve_responses": [f"formal-{step}"],
            "proposal_rewards": [0.25],
            "solve_rewards": [1.0],
            "valid_programs": [f"program-{step}"],
            "program_pool_size": step,
        }
        for step in range(1, 21)
    ]

    assert validate_trace_records(records) == []
    records[0]["difficulty_solve_responses"][0]["responses"].pop()
    errors = validate_trace_records(records)
    assert any("exactly 8" in error for error in errors)


def test_hash_and_jsonl_writer_are_stable(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"b": 2, "a": 1}, {"value": "测试"}])

    assert path.read_text(encoding="utf-8") == ('{"a":1,"b":2}\n{"value":"测试"}\n')
    assert sha256_file(path) == (
        "51101541a687f6ab84f84c784b63932e3d6761aaa108e4ba31840a4223b1b205"
    )
