#!/usr/bin/env python3
"""Precompute human-readable subtasks for all questions in a JSONL dataset.

For each question, reads the GT to extract columns + entities, then generates
natural-language subtasks grouped by row at different worker counts.

Output: a JSON file with structure:
[
  {
    "question": "...",
    "key_column": "...",
    "columns": [...],
    "gt_answer": {...},
    "subtasks_by_cpw": {
      "1": ["Find ... for the entry where ... is ...", ...],
      "2": [...],
      ...
    }
  },
  ...
]

Usage:
    python scripts/precompute_subtasks.py \
        --dataset_path ~/data/width.jsonl \
        --cells_per_worker 1 2 4 8 16 32 \
        --output precomputed_subtasks.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_markdown_table(text: str) -> pd.DataFrame:
    """Parse a markdown table string into a DataFrame."""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    # Filter separator lines
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if set(stripped).issubset(set("|-: ")) or "|" not in stripped:
            continue
        lines.append(line)
    if not lines:
        return None
    import io
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    return df


def _write_discovery_subtask(question, key_column):
    """Generate a human-readable discovery subtask to find all entity names."""
    return (
        f"The main question is:\n{question}\n\n"
        f"Your task: Identify all entities that should appear in the table. "
        f"Find every distinct value for the \"{key_column}\" column based on the question above. "
        f"Use search tools to verify your list.\n\n"
        f"Output the complete list in JSON format:\n"
        f'{{"keys": ["value1", "value2", ...]}}'
    )


def _write_subtask(question, key_column, cell_queries_slice):
    """Generate a single human-readable subtask for cell filling."""
    by_entity = {}
    for cq in cell_queries_slice:
        rk = cq["row_key"]
        if rk not in by_entity:
            by_entity[rk] = []
        by_entity[rk].append(cq["column"])

    entity_lines = []
    for entity, cols in by_entity.items():
        cols_str = ", ".join(cols)
        entity_lines.append(
            f"  - For the entry where {key_column} is \"{entity}\", find: {cols_str}"
        )

    body = "\n".join(entity_lines)
    return (
        f"The main question is:\n{question}\n\n"
        f"Please search for and verify the following information:\n{body}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--cells_per_worker", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--output", default="precomputed_subtasks.json")
    args = p.parse_args()

    with open(args.dataset_path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    results = []
    for idx, sample in enumerate(raw):
        question = sample.get("question") or sample.get("prompt")
        answer_raw = sample.get("answer", "")
        unique_cols = sample.get("unique_columns", [])

        # Parse GT table
        gt_df = parse_markdown_table(answer_raw)
        if gt_df is None:
            print(f"[{idx+1}] SKIP: could not parse GT table")
            continue

        columns = list(gt_df.columns)
        key_column = unique_cols[0].strip() if unique_cols else columns[0]
        if key_column not in columns:
            key_column = columns[0]

        # Entities from GT
        gt_entities = list(gt_df[key_column].astype(str).str.strip().unique())

        # Generate all cell queries
        cell_queries = []
        for entity in gt_entities:
            for col in columns:
                if col == key_column:
                    continue
                cell_queries.append({"row_key": entity, "column": col})

        total_cells = len(cell_queries)
        n_nonkey = len(columns) - 1
        # Prefer grouping by row (one entity = one subtask, n_nonkey cells)
        # then combine rows to reach target cpw

        subtasks_by_cpw = {}
        for cpw in args.cells_per_worker:
            # Group: first try one row per subtask, then combine to reach cpw
            rows_per_worker = max(1, cpw // n_nonkey)
            batches = []
            i = 0
            while i < total_cells:
                end = min(i + rows_per_worker * n_nonkey, total_cells)
                batches.append(cell_queries[i:end])
                i = end

            # Refine: merge small batches to match cpw more closely
            # (simple approach: keep as-is; rows_per_worker already approximates cpw)
            subtasks = [
                _write_subtask(question, key_column, batch)
                for batch in batches
            ]
            subtasks_by_cpw[str(cpw)] = subtasks

        discovery_subtask = _write_discovery_subtask(question, key_column)

        print(f"[{idx+1}/{len(raw)}] {len(gt_entities)} rows × {n_nonkey} cols = "
              f"{total_cells} cells | key={key_column} | question: {question[:80]}...")

        # Store GT answer dict for evaluate_markdown later
        gt_answer_dict = {
            "answer": answer_raw,
            "is_markdown": True,
            "unique_columns": unique_cols,
            "instance_id": sample.get("instance_id", sample.get("id", idx)),
            "language": sample.get("language", "en"),
        }

        results.append({
            "question": question,
            "key_column": key_column,
            "columns": columns,
            "gt_answer": gt_answer_dict,
            "cell_queries": cell_queries,
            "discovery_subtask": discovery_subtask,
            "subtasks_by_cpw": subtasks_by_cpw,
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} precomputed questions to {args.output}")


if __name__ == "__main__":
    main()
