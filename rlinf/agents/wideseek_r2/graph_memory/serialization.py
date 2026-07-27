# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Bounded, JSON-safe graph snapshots for rollout diagnostics."""

from __future__ import annotations

import copy
import dataclasses
import json
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert graph dataclasses/enums into bounded JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            key: to_jsonable(item) for key, item in dataclasses.asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def snapshot_json(runtime: Any, *, max_nodes: int | None = None) -> str:
    """Serialize a bounded runtime snapshot without source excerpts by default."""

    snapshot = copy.deepcopy(runtime.snapshot(max_nodes=max_nodes))
    if not runtime.config.snapshot_include_source_excerpt:
        for node in snapshot.get("evidence_nodes", []):
            if getattr(node, "kind", None) and str(node.kind.value) == "source":
                for key in ("excerpt", "content", "text", "body"):
                    node.payload.pop(key, None)
    return json.dumps(to_jsonable(snapshot), ensure_ascii=False, sort_keys=True)
