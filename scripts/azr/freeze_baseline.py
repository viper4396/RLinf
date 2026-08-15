#!/usr/bin/env python3
"""Freeze and validate the Phase 0 AZR reproduction contract.

This tool deliberately does not launch training. It records the exact source
revisions, configuration, input hashes, environment metadata, and fixed CPU
examples that a short paper-version AZR run must use. A strict validation is
intended to be run immediately before allocating GPUs.

The generated manifest distinguishes a frozen contract from a completed short
run. Missing external artifacts (the validation parquet or model checkpoint,
for example) are recorded explicitly instead of being silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

AZR_PAPER_COMMIT = "41ed983cdf541cfcd2f963f33c055d50074f3c90"
RLINF_BASELINE_COMMIT = "a099ff4a2d46ca1fa4d15cf10164a180fbecad16"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B"
PROBLEM_TYPES = ("code_i", "code_o", "code_f")
DEFAULT_FIXED_COUNTS = {"code_i": 20, "code_o": 20, "code_f": 10}
TRACE_REQUIRED_FIELDS = (
    "global_step",
    "proposal_responses",
    "difficulty_solve_responses",
    "formal_solve_responses",
    "proposal_rewards",
    "solve_rewards",
    "valid_programs",
    "program_pool_size",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "examples/reasoning/config/absolute_zero"
_AZR_SOURCE_FILES = (
    "scripts/selfplay/coder3b.sh",
    "absolute_zero_reasoner/configs/azr_ppo_trainer.yaml",
)
_SEED_FILES = {
    "io": "data/3b_coder_seed_io.jsonl",
    "error": "data/3b_coder_error_seed_io.jsonl",
    "code_f": "data/3b_coder_code_f_seed_io.jsonl",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or directory deterministically.

    Directory hashes include each relative file name and file digest in sorted
    order. This makes a model checkpoint directory auditable without relying
    on filesystem traversal order.
    """
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_name = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative_name)
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _git_output(repo: Path, *arguments: str) -> str | None:
    """Return one git command's trimmed stdout, or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_file_sha256(repo: Path, revision: str, relative_path: str) -> str | None:
    """Hash a file exactly as stored in a git revision."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{revision}:{relative_path}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _git_repository_record(
    repo: Path,
    *,
    name: str,
    expected_commit: str,
    reference: str,
    clean_required: bool,
) -> dict[str, Any]:
    """Build a source revision record for the manifest."""
    resolved_reference = _git_output(repo, "rev-parse", f"{reference}^{{commit}}")
    working_tree_commit = _git_output(repo, "rev-parse", "HEAD")
    status = _git_output(repo, "status", "--porcelain")
    dirty = status is None or bool(status)
    return {
        "name": name,
        "path": str(repo),
        "reference": reference,
        "expected_commit": expected_commit,
        "resolved_reference_commit": resolved_reference,
        "working_tree_commit": working_tree_commit,
        "working_tree_dirty": dirty,
        "clean_required": clean_required,
        "matches_expected_commit": resolved_reference == expected_commit,
    }


def _artifact_record(
    path: Path | None,
    *,
    required: bool,
    relative_path: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Describe and hash one input artifact without hiding missing files."""
    record: dict[str, Any] = {
        "required": required,
        "relative_path": relative_path,
        "observed_path": str(path) if path is not None else None,
        "exists": bool(path and path.exists()),
        "sha256": None,
        "expected_sha256": expected_sha256,
        "matches_expected": None,
    }
    if path is not None and path.exists():
        record["sha256"] = sha256_path(path)
        record["size_bytes"] = (
            path.stat().st_size
            if path.is_file()
            else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        )
        if expected_sha256 is not None:
            record["matches_expected"] = record["sha256"] == expected_sha256
    return record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and reject malformed or non-object records."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def build_fixed_examples(
    io_records: Iterable[Mapping[str, Any]],
    code_f_records: Iterable[Mapping[str, Any]],
    *,
    counts: Mapping[str, int] | None = None,
    io_source: str = "data/3b_coder_seed_io.jsonl",
    code_f_source: str = "data/3b_coder_code_f_seed_io.jsonl",
) -> list[dict[str, Any]]:
    """Select deterministic fixed examples for the three AZR task types.

    The paper seed file is used in source order. The same IO records are
    intentionally represented once as ``code_i`` and once as ``code_o``: the
    task type changes the prompt and evaluator while preserving the reference
    program and input/output pair.
    """
    requested = dict(counts or DEFAULT_FIXED_COUNTS)
    if set(requested) != set(PROBLEM_TYPES):
        raise ValueError(f"counts must contain exactly {PROBLEM_TYPES}")
    if any(value < 0 for value in requested.values()):
        raise ValueError("fixed example counts must be non-negative")

    io_items = list(io_records)
    code_f_items = list(code_f_records)
    if len(io_items) < max(requested["code_i"], requested["code_o"]):
        raise ValueError("the IO seed dataset does not contain enough examples")
    if len(code_f_items) < requested["code_f"]:
        raise ValueError("the code_f seed dataset does not contain enough examples")

    examples: list[dict[str, Any]] = []
    for problem_type in ("code_i", "code_o"):
        for source_index, payload in enumerate(io_items[: requested[problem_type]]):
            examples.append(
                {
                    "example_id": f"{problem_type}-{source_index:03d}",
                    "problem_type": problem_type,
                    "source": {"path": io_source, "index": source_index},
                    "payload": payload,
                }
            )
    for source_index, payload in enumerate(code_f_items[: requested["code_f"]]):
        examples.append(
            {
                "example_id": f"code_f-{source_index:03d}",
                "problem_type": "code_f",
                "source": {"path": code_f_source, "index": source_index},
                "payload": payload,
            }
        )
    return examples


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write JSON objects in a stable, one-record-per-line format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            )
            file.write("\n")


def validate_trace_records(
    records: Iterable[Mapping[str, Any]],
    *,
    minimum_steps: int = 20,
    maximum_steps: int = 100,
) -> list[str]:
    """Validate the per-step trace needed to reconstruct an AZR short run.

    The trace stores difficulty responses grouped by proposal rather than as a
    flat list. This makes it possible to detect cross-problem mixing and to
    verify that every valid proposal received exactly eight difficulty samples.
    """
    items = list(records)
    errors: list[str] = []
    if not minimum_steps <= len(items) <= maximum_steps:
        errors.append(
            f"trace must contain {minimum_steps}-{maximum_steps} steps, got {len(items)}"
        )

    seen_steps: set[int] = set()
    for line_number, record in enumerate(items, start=1):
        missing_fields = [
            field for field in TRACE_REQUIRED_FIELDS if field not in record
        ]
        if missing_fields:
            errors.append(f"trace line {line_number} is missing {missing_fields}")
            continue

        global_step = record["global_step"]
        if not isinstance(global_step, int) or isinstance(global_step, bool):
            errors.append(f"trace line {line_number} has a non-integer global_step")
        elif global_step in seen_steps:
            errors.append(f"trace line {line_number} repeats global_step {global_step}")
        else:
            seen_steps.add(global_step)

        proposals = record["proposal_responses"]
        if not isinstance(proposals, list):
            errors.append(f"trace line {line_number} proposal_responses must be a list")
            proposal_ids: set[str] = set()
        else:
            proposal_ids = set()
            for proposal in proposals:
                if not isinstance(proposal, Mapping):
                    errors.append(f"trace line {line_number} has a non-object proposal")
                    continue
                proposal_id = proposal.get("proposal_id")
                if not isinstance(proposal_id, str) or not proposal_id:
                    errors.append(
                        f"trace line {line_number} has a proposal without an id"
                    )
                elif proposal_id in proposal_ids:
                    errors.append(
                        f"trace line {line_number} repeats proposal {proposal_id}"
                    )
                else:
                    proposal_ids.add(proposal_id)

        difficulty_groups = record["difficulty_solve_responses"]
        if not isinstance(difficulty_groups, list):
            errors.append(
                f"trace line {line_number} difficulty_solve_responses must be a list"
            )
        else:
            difficulty_ids: set[str] = set()
            for group in difficulty_groups:
                if not isinstance(group, Mapping):
                    errors.append(
                        f"trace line {line_number} has a non-object difficulty group"
                    )
                    continue
                proposal_id = group.get("proposal_id")
                responses = group.get("responses")
                if proposal_id not in proposal_ids:
                    errors.append(
                        f"trace line {line_number} difficulty group references unknown "
                        f"proposal {proposal_id!r}"
                    )
                if proposal_id in difficulty_ids:
                    errors.append(
                        f"trace line {line_number} repeats difficulty group {proposal_id}"
                    )
                if not isinstance(responses, list) or len(responses) != 8:
                    errors.append(
                        f"trace line {line_number} proposal {proposal_id!r} must have "
                        "exactly 8 difficulty responses"
                    )
                if isinstance(proposal_id, str):
                    difficulty_ids.add(proposal_id)

        for field in (
            "formal_solve_responses",
            "proposal_rewards",
            "solve_rewards",
            "valid_programs",
        ):
            if not isinstance(record[field], list):
                errors.append(f"trace line {line_number} {field} must be a list")
        pool_size = record["program_pool_size"]
        if (
            not isinstance(pool_size, int)
            or isinstance(pool_size, bool)
            or pool_size < 0
        ):
            errors.append(
                f"trace line {line_number} program_pool_size must be non-negative"
            )
    return errors


def _package_versions() -> dict[str, str | None]:
    """Collect versions relevant to the AZR baseline when installed."""
    packages = {
        "datasets": "datasets",
        "hydra_core": "hydra-core",
        "numpy": "numpy",
        "omegaconf": "omegaconf",
        "pyyaml": "PyYAML",
        "ray": "ray",
        "torch": "torch",
        "transformers": "transformers",
        "vllm": "vllm",
    }
    versions: dict[str, str | None] = {}
    for key, distribution in packages.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _missing_required_artifacts(artifacts: Mapping[str, Any]) -> list[str]:
    """Return required artifact names that are absent or hash-mismatched."""
    missing: list[str] = []
    for name, record in artifacts.items():
        if not isinstance(record, Mapping):
            missing.append(name)
            continue
        if record.get("required") and not record.get("exists"):
            missing.append(name)
        if record.get("required") and record.get("matches_expected") is False:
            missing.append(f"{name} (sha256 mismatch)")
    return missing


def build_manifest(
    *,
    azr_root: Path,
    rlinf_root: Path,
    config_path: Path,
    fixed_examples_path: Path,
    validation_data: Path | None = None,
    model_checkpoint: Path | None = None,
    baseline_trace: Path | None = None,
    fixed_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build the Phase 0 manifest and return it as a Python mapping."""
    azr_root = azr_root.expanduser().resolve()
    rlinf_root = rlinf_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    fixed_examples_path = fixed_examples_path.expanduser().resolve()
    validation_data = (
        validation_data.expanduser().resolve()
        if validation_data is not None
        else azr_root / "data/code_reason/test_answer.parquet"
    )

    io_seed = azr_root / _SEED_FILES["io"]
    code_f_seed = azr_root / _SEED_FILES["code_f"]
    error_seed = azr_root / _SEED_FILES["error"]
    counts = dict(fixed_counts or DEFAULT_FIXED_COUNTS)

    fixed_examples = build_fixed_examples(
        _read_jsonl(io_seed),
        _read_jsonl(code_f_seed),
        counts=counts,
        io_source=_SEED_FILES["io"],
        code_f_source=_SEED_FILES["code_f"],
    )
    write_jsonl(fixed_examples_path, fixed_examples)

    expanded_config = _load_yaml(config_path)
    azr_source = _git_repository_record(
        azr_root,
        name="AZR",
        expected_commit=AZR_PAPER_COMMIT,
        reference="origin/paper",
        clean_required=True,
    )
    rlinf_source = _git_repository_record(
        rlinf_root,
        name="RLinf",
        expected_commit=RLINF_BASELINE_COMMIT,
        reference=RLINF_BASELINE_COMMIT,
        clean_required=True,
    )

    source_files = []
    for relative_path in _AZR_SOURCE_FILES:
        source_files.append(
            {
                "path": relative_path,
                "sha256": git_file_sha256(azr_root, AZR_PAPER_COMMIT, relative_path),
            }
        )

    artifacts: dict[str, Any] = {
        "seed_io": _artifact_record(
            io_seed,
            required=True,
            relative_path=_SEED_FILES["io"],
            expected_sha256=git_file_sha256(
                azr_root, AZR_PAPER_COMMIT, _SEED_FILES["io"]
            ),
        ),
        "seed_error": _artifact_record(
            error_seed,
            required=False,
            relative_path=_SEED_FILES["error"],
            expected_sha256=git_file_sha256(
                azr_root, AZR_PAPER_COMMIT, _SEED_FILES["error"]
            ),
        ),
        "seed_code_f": _artifact_record(
            code_f_seed,
            required=True,
            relative_path=_SEED_FILES["code_f"],
            expected_sha256=git_file_sha256(
                azr_root, AZR_PAPER_COMMIT, _SEED_FILES["code_f"]
            ),
        ),
        "validation_dataset": _artifact_record(
            validation_data,
            required=True,
            relative_path="data/code_reason/test_answer.parquet",
        ),
        "model_checkpoint": _artifact_record(model_checkpoint, required=True),
        "fixed_examples": _artifact_record(
            fixed_examples_path,
            required=True,
            relative_path=fixed_examples_path.name,
        ),
        "short_run_trace": _artifact_record(
            baseline_trace,
            required=True,
            relative_path="baseline_trace.jsonl",
        ),
    }

    trace_records: list[dict[str, Any]] = []
    trace_errors: list[str] = []
    if baseline_trace is not None and baseline_trace.exists():
        try:
            trace_records = _read_jsonl(baseline_trace)
            trace_errors = validate_trace_records(trace_records)
        except ValueError as error:
            trace_errors = [str(error)]

    missing_artifacts = _missing_required_artifacts(artifacts)
    repository_blockers = [
        f"{record['name']} worktree is dirty"
        for record in (azr_source, rlinf_source)
        if record["clean_required"] and record["working_tree_dirty"]
    ]
    repository_blockers.extend(
        f"{record['name']} reference does not resolve to the expected commit"
        for record in (azr_source, rlinf_source)
        if not record["matches_expected_commit"]
    )
    blocking_items = repository_blockers + list(missing_artifacts)
    blocking_items.extend(f"short_run_trace ({error})" for error in trace_errors)
    fixed_counts_observed = {
        problem_type: sum(
            example["problem_type"] == problem_type for example in fixed_examples
        )
        for problem_type in PROBLEM_TYPES
    }
    short_run_complete = bool(trace_records) and not trace_errors
    short_run_status = "recorded" if short_run_complete else "not_run"
    if baseline_trace is not None and baseline_trace.exists() and trace_errors:
        short_run_status = "invalid"

    return {
        "schema_version": 1,
        "phase": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "baseline_pending" if blocking_items else "frozen",
        "contract": {
            "name": "azr-paper-reinforce-plus-plus",
            "behavior_source": "AZR origin/paper@41ed983",
            "training_semantics": {
                "advantage": "reinforce_plus_plus",
                "critic": False,
                "policy_loss": "ppo_clipped_actor",
                "proposal_rollout_n": 1,
                "difficulty_rollout_n": 8,
                "formal_solve_rollout_n": 1,
                "difficulty_trajectories_trainable": False,
                "problem_types": list(PROBLEM_TYPES),
            },
            "comparison_rule": (
                "Differences must be attributable to code, environment, or random sampling."
            ),
        },
        "repositories": {
            "azr": {**azr_source, "source_files": source_files},
            "rlinf": rlinf_source,
        },
        "model": {
            "id": MODEL_ID,
            "checkpoint": dict(artifacts["model_checkpoint"]),
            "revision": None,
            "revision_must_be_frozen_before_run": True,
        },
        "configuration": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "resolved": expanded_config,
        },
        "artifacts": artifacts,
        "fixed_examples": {
            "path": str(fixed_examples_path),
            "sha256": artifacts["fixed_examples"]["sha256"],
            "requested_counts": counts,
            "observed_counts": fixed_counts_observed,
            "minimum_total": 50,
        },
        "trace_schema": {
            "version": 1,
            "format": "jsonl",
            "one_record_per_global_step": True,
            "difficulty_group_size": 8,
            "difficulty_group_key": "proposal_id",
            "difficulty_responses_key": "responses",
            "trainable_trajectory_groups": [
                "proposal_responses",
                "formal_solve_responses",
            ],
            "excluded_trajectory_group": "difficulty_solve_responses",
        },
        "short_baseline": {
            "status": short_run_status,
            "requested_step_range": {"minimum": 20, "maximum": 100},
            "trace": artifacts["short_run_trace"],
            "trace_errors": trace_errors,
            "executed_steps": len(trace_records) if trace_records else None,
            "required_per_step_fields": list(TRACE_REQUIRED_FIELDS),
            "difficulty_group_size": 8,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
            "relevant_env": {
                "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
                "RAY_memory_monitor_refresh_ms": os.environ.get(
                    "RAY_memory_monitor_refresh_ms"
                ),
                "RAY_LOGGING_LEVEL": os.environ.get("RAY_LOGGING_LEVEL"),
                "HYDRA_FULL_ERROR": os.environ.get("HYDRA_FULL_ERROR"),
            },
        },
        "acceptance": {
            "source_commits_frozen": all(
                record["matches_expected_commit"]
                and not (record["clean_required"] and record["working_tree_dirty"])
                for record in (azr_source, rlinf_source)
            ),
            "expanded_configuration_frozen": bool(expanded_config),
            "fixed_examples_cover_all_tasks": fixed_counts_observed == counts,
            "required_artifact_hashes_recorded": not missing_artifacts,
            "short_run_20_to_100_steps_recorded": short_run_complete,
            "reproduction_contract_complete": not blocking_items,
        },
        "blocking_artifacts": missing_artifacts,
        "blocking_repositories": repository_blockers,
        "blocking_items": blocking_items,
    }


def validate_manifest(
    manifest: Mapping[str, Any], *, strict: bool = False
) -> list[str]:
    """Return contract violations found in a generated manifest."""
    errors: list[str] = []
    contract = manifest.get("contract", {})
    semantics = contract.get("training_semantics", {})
    if contract.get("behavior_source") != "AZR origin/paper@41ed983":
        errors.append("behavior source is not the paper AZR commit")
    if semantics.get("advantage") != "reinforce_plus_plus":
        errors.append("advantage must be reinforce_plus_plus")
    if semantics.get("critic") is not False:
        errors.append("critic must be disabled")
    if semantics.get("difficulty_trajectories_trainable") is not False:
        errors.append("difficulty trajectories must be excluded from training")
    if semantics.get("problem_types") != list(PROBLEM_TYPES):
        errors.append(f"problem types must be {list(PROBLEM_TYPES)}")

    repositories = manifest.get("repositories", {})
    repository_expectations = {
        "azr": (AZR_PAPER_COMMIT, "AZR", "41ed983"),
        "rlinf": (RLINF_BASELINE_COMMIT, "RLinf", "a099ff4a"),
    }
    for key, (expected, name, short_commit) in repository_expectations.items():
        repository = repositories.get(key, {})
        if repository.get("expected_commit") != expected:
            errors.append(f"{name} baseline commit is not frozen to {short_commit}")
        if repository.get("clean_required") and repository.get("working_tree_dirty"):
            errors.append(f"{name} baseline worktree is dirty")
    if manifest.get("model", {}).get("id") != MODEL_ID:
        errors.append(f"model must be {MODEL_ID}")

    fixed_examples = manifest.get("fixed_examples", {})
    requested = fixed_examples.get("requested_counts")
    observed = fixed_examples.get("observed_counts")
    if requested != observed:
        errors.append("fixed example counts do not match the requested coverage")
    if isinstance(observed, Mapping):
        if sum(observed.values()) < 50:
            errors.append("at least 50 fixed examples are required")
    else:
        errors.append("fixed example counts are missing")

    artifacts = manifest.get("artifacts", {})
    errors.extend(
        f"missing artifact: {item}" for item in _missing_required_artifacts(artifacts)
    )
    short_baseline = manifest.get("short_baseline", {})
    errors.extend(
        f"invalid short-run trace: {error}"
        for error in short_baseline.get("trace_errors", [])
    )
    if strict:
        acceptance = manifest.get("acceptance", {})
        if not acceptance.get("short_run_20_to_100_steps_recorded", False):
            errors.append("the 20-100 step AZR short run has not been recorded")
        if manifest.get("status") not in {"frozen", "complete"}:
            errors.append("manifest is not marked frozen or complete")
    return errors


def _parse_counts(raw_counts: str) -> dict[str, int]:
    """Parse ``code_i=20,code_o=20,code_f=10`` CLI syntax."""
    counts: dict[str, int] = {}
    for item in raw_counts.split(","):
        key, separator, raw_value = item.partition("=")
        if not separator or key not in PROBLEM_TYPES:
            raise argparse.ArgumentTypeError(f"invalid fixed count: {item}")
        try:
            counts[key] = int(raw_value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid fixed count: {item}") from error
    if set(counts) != set(PROBLEM_TYPES):
        raise argparse.ArgumentTypeError(
            f"counts must contain {','.join(PROBLEM_TYPES)}"
        )
    return counts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--azr-root", type=Path, default=Path("~/AZR").expanduser())
    parser.add_argument("--rlinf-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument(
        "--baseline-trace",
        type=Path,
        help="JSONL trace with one reconstructable record per short-run step",
    )
    parser.add_argument(
        "--fixed-counts",
        type=_parse_counts,
        default=dict(DEFAULT_FIXED_COUNTS),
        help="fixed example counts, e.g. code_i=20,code_o=20,code_f=10",
    )
    parser.add_argument("--validate-only", type=Path, metavar="MANIFEST")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate or validate a Phase 0 manifest."""
    args = _build_parser().parse_args(argv)
    if args.validate_only is not None:
        manifest = _load_yaml(args.validate_only.expanduser().resolve())
        errors = validate_manifest(manifest, strict=args.strict)
        if errors:
            print("Baseline manifest validation failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print(f"Baseline manifest is valid: {args.validate_only}")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "baseline_config.yaml"
    fixed_examples_path = output_dir / "fixed_examples.jsonl"
    manifest_path = output_dir / "baseline_manifest.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"expanded baseline config is missing: {config_path}; add it before freezing"
        )

    manifest = build_manifest(
        azr_root=args.azr_root,
        rlinf_root=args.rlinf_root,
        config_path=config_path,
        fixed_examples_path=fixed_examples_path,
        validation_data=args.validation_data,
        model_checkpoint=args.model_checkpoint,
        baseline_trace=args.baseline_trace,
        fixed_counts=args.fixed_counts,
    )
    with manifest_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, allow_unicode=True, sort_keys=False)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {fixed_examples_path}")
    if manifest["blocking_items"]:
        print("Baseline remains pending; blocking items:")
        for item in manifest["blocking_items"]:
            print(f"- {item}")
    if args.strict:
        errors = validate_manifest(manifest, strict=True)
        if errors:
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
