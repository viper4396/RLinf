#!/usr/bin/env python3
"""Cell-level Oracle experiment: measure F1 vs workers-per-turn.

Flow:
1. Load original JSONL (for GT) + discovery_subtasks JSONL.
2. Discovery worker: policy model resolves discovery_subtask → entity list.
3. Judge LLM: generates human-readable cell subtasks from discovered entities.
4. Cell workers: policy model resolves each subtask → cell values.
5. Judge LLM: evaluate_markdown → F1.

Usage:
    python scripts/cell_oracle_experiment.py \
        --dataset_path ~/data/width.jsonl \
        --discovery_path ~/data/discovery_subtasks.jsonl \
        --num_samples 128 \
        --model_path /path/to/Qwen3-4B \
        --sglang_url http://xx:30000 \
        --judge_sglang_url http://xx:30001 \
        --search_server xx:8000 \
        --cells_per_worker 1 2 4 8 16 32 \
        --max_concurrent_workers 64
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Add repo root so we can import rlinf modules.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Helpers: lightweight LLM call using raw HTTP (avoids Ray / agent-loop deps).
# ---------------------------------------------------------------------------
async def _http_json_post(session, url: str, payload: dict, max_retries: int = 3) -> dict:
    import aiohttp
    import asyncio as _asyncio

    last_err = None
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, timeout=300) as resp:
                return await resp.json()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                await _asyncio.sleep(wait)
    raise last_err


async def sglang_generate(
    session,
    sglang_url: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """Call SGLang chat completions API."""
    url = f"{sglang_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = await _http_json_post(session, url, payload)
    return data["choices"][0]["message"]["content"]


async def sglang_complete(
    session,
    sglang_url: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    stop: Optional[list[str]] = None,
) -> str:
    """Call SGLang completions API with raw prompt text."""
    url = f"{sglang_url.rstrip('/')}/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop
    data = await _http_json_post(session, url, payload)
    return data["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Search / access client  (reuses RLinf classes when possible).
# ---------------------------------------------------------------------------
def _make_search_client(
    online: bool, server_addr: Optional[str], topk: int = 5
):
    """Create a search/access client matching the tool worker."""
    from omegaconf import OmegaConf

    base = {
        "tools": {
            "search": {
                "server_addr": server_addr or "127.0.0.1:8000",
                "topk": topk,
            }
        }
    }
    if online:
        from rlinf.agents.wideseek_r1.tools import AsyncOnlineSearchClient

        base["tools"]["search"]["serper_api_key"] = os.environ.get(
            "SERPER_API_KEY", ""
        )
        base["tools"]["search"]["jina_api_key"] = os.environ.get(
            "JINA_API_KEY", ""
        )
        return AsyncOnlineSearchClient(OmegaConf.create(base))
    else:
        from rlinf.agents.wideseek_r1.tools import AsyncSearchClient

        return AsyncSearchClient(OmegaConf.create(base))


# ---------------------------------------------------------------------------
# Discovery phase: find entity names for the question.
# ---------------------------------------------------------------------------
DISCOVERY_PROMPT = """# Role
You are a research assistant. Given a question that expects a markdown table as answer, your job is to identify all values for the primary key column "{key_column}" that should appear in the table. Use search and access tools to verify your list.

You must conduct reasoning inside <think> and </think> first every time you get new information.

# Final Answer
When you have collected enough information, output the complete list of key values in JSON format:

```json
{{
  "keys": ["value1", "value2", ...]
}}

# Rules
- List every distinct value for the column "{key_column}".
- Use the search and access tools to find reliable sources.
- Output canonical/official names or values.
- Do not include duplicates."""


async def discover_entities(
    session,
    sglang_url: str,
    question: str,
    key_column: str,
    search_client,
    tokenizer,
    toolcall_parser,
    tools_description_en,
    max_entities: int = 50,
    max_turns: int = 10,
) -> list[str]:
    """Run the discovery phase using real tool-call format with search/access."""
    messages = [
        {"role": "system", "content": DISCOVERY_PROMPT.format(
            question=question, key_column=key_column,
        )},
        {"role": "user", "content": f"Find all distinct values for the key column '{key_column}' in the table described by: {question}"},
    ]
    tools = [
        tools_description_en["search"],
        tools_description_en["access"],
    ]

    for turn in range(max_turns):
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, tools=tools
        )
        prompt_ids = prompt_ids[-8192:]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        resp = await sglang_complete(session, sglang_url, prompt_text,
            max_tokens=2048, stop=["<|im_end|>", "<|endoftext|>"],
        )
        # print(f"    [Discovery turn {turn+1}] {len(resp)} chars: {resp[:200]}...")

        parsed = await toolcall_parser(resp, role="worker", max_toolcall_per_worker=5)
        tool_requests = parsed[1] if isinstance(parsed, tuple) else []

        if not tool_requests:
            messages.append({"role": "assistant", "content": resp})
            break

        search_requests = [r for r in tool_requests if r.name == "search"]
        access_requests = [r for r in tool_requests if r.name == "access"]

        snippets, urls_to_access = [], []
        if search_requests:
            queries = [r.arguments.get("query", "") for r in search_requests]
            try:
                raw = await search_client.query_async({"queries": queries, "topk": 5})
            except Exception as e:
                print(f"    [Discovery] Search error: {e}")
                raw = []
            for batch in raw:
                for doc, url in zip(batch.get("documents", []), batch.get("urls", [])):
                    if url:
                        urls_to_access.append(url)
                        snippets.append(f"- [{url[:80]}]: {str(doc)[:200]}")

        for r in access_requests:
            url = r.arguments.get("url", "")
            if url and url not in urls_to_access:
                urls_to_access.append(url)

        access_text = ""
        if urls_to_access:
            try:
                pages = await search_client.access_async(urls_to_access[:3])
                access_text = "\n\n".join(
                    f"URL: {u}\n{p.get('page', '')[:3000]}"
                    for u, p in zip(urls_to_access[:3], pages)
                )
            except Exception as e:
                print(f"    [Discovery] Access error: {e}")

        result = "Search results:\n" + "\n".join(snippets) if snippets else "(no search)"
        if access_text:
            result += f"\n\nPage content:\n{access_text}"

        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "tool", "content": result})

    # Final extraction.
    messages.append({"role": "user", "content": "Output the complete list of key values in JSON format."})
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    prompt_ids = prompt_ids[-8192:]
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    final_text = await sglang_complete(session, sglang_url, prompt_text,
        max_tokens=4096, stop=["<|im_end|>", "<|endoftext|>"],
    )

    try:
        import re
        pat = r"```json\s*(.*?)\s*```"
        matches = re.findall(pat, final_text, re.DOTALL)
        data = json.loads(matches[-1] if matches else final_text)
        keys = data.get("keys", [])
        return [str(k).strip() for k in keys if k][:max_entities]
    except Exception:
        # Fallback: parse line-by-line.
        lines = [l.strip("- 0123456789. ") for l in final_text.strip().split("\n")]
        return [l for l in lines if l and len(l) > 1][:max_entities]


# ---------------------------------------------------------------------------
# Cell query generation.
# ---------------------------------------------------------------------------
def generate_cell_queries(
    entities: list[str], columns: list[str], key_column: str
) -> list[dict]:
    """Return one dict per cell: {row_key, column, query_text}."""
    queries = []
    for entity in entities:
        for col in columns:
            if col == key_column:
                continue
            queries.append(
                {
                    "row_key": entity,
                    "column": col,
                    "query": f"What is {entity}'s {col}?",
                }
            )
    return queries


# ---------------------------------------------------------------------------
# Single-cell worker: search + access + extract.
# ---------------------------------------------------------------------------
WORKER_SYSTEM_PROMPT = """# Role
You are a sub-agent responsible for a specific part of a larger task. Your job is to complete your assigned subtask accurately using search and access tools with detailed evidence. You are not expected to solve the main task as a whole.

You must conduct reasoning inside <think> and </think> first every time you get new information.

# Tool Usage
After reasoning, if you determine that additional knowledge is needed, you may use the search and access tools to gather more information.

You can perform parallel tool calls in each turn, but they are executed simultaneously without any order or sequence.

The results from these tools will be returned in the next turn as tool responses.

Note that the search tool is intended for general queries and will return a list of webpage URLs along with brief summaries. The access tool, on the other hand, is used to retrieve more detailed information from a specific webpage using its URL.

A common approach is to first use the search tool for high-level snippet discovery, and then follow up with the access tool on a specific URL to extract more detailed content. Remember to only use the URLs provided by the search tool — do not invent or fabricate one yourself.

You can perform multiple turns of tool calls. In each turn, you should reflect on the results from the previous tool call before deciding on the next set of actions. Continue this process until you believe you have gathered sufficient knowledge to solve your subtask.

# Final Answer
When you have collected enough reliable information, output your verified findings in JSON format:

```json
{{
  "findings": [
    {{"row_key": "...", "column": "...", "value": "..."}},
    ...
  ]
}}

# Rules for JSON output
- Only report values you are confident about from a reliable source.
- If you cannot find a value, do NOT include that cell in your output.
- CRITICAL: Keep the row_key exactly as provided in the query. Do NOT modify, concatenate, or embellish it.
- The "value" field should be exactly as it appears in the source."""


async def run_one_cell_worker(
    session,
    sglang_url: str,
    search_client,
    subtask_text: str,
    tokenizer,
    toolcall_parser,
    tools_description_en,
    max_turns: int = 20,
    output_mode: str = "findings",
    print_final: bool = False,
) -> list[dict]:
    """Run a worker using the real tool-call format (<tool_call> JSON).

    *output_mode*: ``"findings"`` → cell values, ``"keys"`` → entity keys (discovery).
    """
    if output_mode == "keys":
        sys_prompt = WORKER_SYSTEM_PROMPT.replace(
            '"findings": [\n    {"row_key": "...", "column": "...", "value": "..."}\n  ]',
            '"keys": ["value1", "value2", ...]'
        ).replace(
            '- CRITICAL: Keep the row_key exactly as provided in the query. Do NOT modify, concatenate, or embellish it.',
            '- Output only the key values, one per entry.'
        )
    else:
        sys_prompt = WORKER_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": subtask_text},
    ]
    tools = [
        tools_description_en["search"],
        tools_description_en["access"],
    ]

    for turn in range(max_turns):
        # Apply chat template with tools → get prompt text.
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, tools=tools
        )
        prompt_ids = prompt_ids[-8192:]  # truncate to avoid context overflow
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # Generate (raw text — template already applied).
        resp = await sglang_complete(session, sglang_url, prompt_text,
            max_tokens=2048,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        # print(f"    [Worker turn {turn+1}] {len(resp)} chars: {resp[:250]}...")

        # Parse tool calls.
        parsed_requests = await toolcall_parser(
            resp, role="worker", max_toolcall_per_worker=5,
        )
        tool_requests = parsed_requests[1] if isinstance(parsed_requests, tuple) else []

        if not tool_requests:
            # No tool calls → model is done, try to extract findings.
            messages.append({"role": "assistant", "content": resp})
            break

        # Execute tools.
        search_requests = [r for r in tool_requests if r.name == "search"]
        access_requests = [r for r in tool_requests if r.name == "access"]

        # Run searches.
        search_snippets = []
        urls_to_access = []
        if search_requests:
            queries = [r.arguments.get("query", "") for r in search_requests]
            # print(f"    [Worker turn {turn+1}] Searching {len(queries)} queries...")
            try:
                search_results_raw = await search_client.query_async(
                    {"queries": queries, "topk": 5}
                )
            except Exception as e:
                print(f"    [Worker] Search error: {e}")
                search_results_raw = []
            for batch in search_results_raw:
                docs = batch.get("documents", [])
                urls = batch.get("urls", [])
                for doc, url in zip(docs, urls):
                    if url:
                        urls_to_access.append(url)
                        search_snippets.append(f"- [{url[:80]}]: {str(doc)[:200]}")

        # Run accesses (merge explicit access requests + URLs from search).
        for r in access_requests:
            url = r.arguments.get("url", "")
            if url and url not in urls_to_access:
                urls_to_access.append(url)

        access_text = ""
        if urls_to_access:
            # print(f"    [Worker turn {turn+1}] Accessing {len(urls_to_access[:3])} URLs...")
            try:
                pages = await search_client.access_async(urls_to_access[:3])
                access_text = "\n\n".join(
                    f"URL: {u}\n{p.get('page', '')[:3000]}"
                    for u, p in zip(urls_to_access[:3], pages)
                )
            except Exception as e:
                print(f"    [Worker] Access error: {e}")

        # Format tool response.
        tool_result = "Search results:\n" + "\n".join(search_snippets) if search_snippets else "(no search)"
        if access_text:
            tool_result += f"\n\nPage content:\n{access_text}"

        messages.append({"role": "assistant", "content": resp})
        messages.append({"role": "tool", "content": tool_result})

    # Final extraction.
    if output_mode == "keys":
        messages.append({
            "role": "user",
            "content": "Output the complete list of key values in JSON format.",
        })
    else:
        messages.append({
            "role": "user",
            "content": "Please output your final verified findings in JSON format.",
        })
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    prompt_ids = prompt_ids[-8192:]
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    final_text = await sglang_complete(session, sglang_url, prompt_text,
        max_tokens=4096,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Parse JSON.
    if print_final:
        print(f"    [Worker] === FINAL OUTPUT ===\n{final_text}\n    [Worker] === END ===")
    try:
        import re
        # Try ```json code block first.
        pat = r"```json\s*(.*?)\s*```"
        matches = re.findall(pat, final_text, re.DOTALL)
        if matches:
            data = json.loads(matches[-1])
        else:
            # Try to find a JSON object anywhere in the text.
            obj_pat = r'\{[^{}]*"keys"\s*:\s*\[.*?\][^{}]*\}'
            obj_matches = re.findall(obj_pat, final_text, re.DOTALL)
            if obj_matches:
                data = json.loads(obj_matches[-1])
            else:
                data = json.loads(final_text)

        if output_mode == "keys":
            keys = data.get("keys", [])
            # Fallback: try extracting from findings format.
            if not keys:
                for f in data.get("findings", []):
                    rk = f.get("row_key", "")
                    if rk and rk not in keys:
                        keys.append(rk)
            return [{"row_key": str(k).strip(), "column": "", "value": ""} for k in keys if k]
        else:
            findings = data.get("findings", [])
            return [
                {"row_key": f["row_key"], "column": f["column"], "value": str(f["value"])}
                for f in findings
                if all(k in f for k in ("row_key", "column", "value"))
            ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main experiment.
# ---------------------------------------------------------------------------
async def run_experiment(
    session,
    sglang_url: str,
    search_client,
    question_data: dict,
    discovery_entry: dict,
    cells_per_worker_values: list[int],
    max_concurrent_workers: int = 8,
    judge_sglang_url: Optional[str] = None,
    tokenizer=None,
    toolcall_parser=None,
    tools_description_en=None,
) -> dict:
    """Run cell-oracle experiment for one question.

    1. Discovery worker → entity list.
    2. Judge LLM generates human-readable subtasks per cpw.
    3. Cell workers → cell values.
    4. evaluate_markdown → F1.

    Returns dict mapping cells_per_worker → (n_workers, f1_score, cells_written).
    """
    question = question_data["prompt"]
    ground_truth = question_data["answer"]
    discovery_subtask = discovery_entry.get("discovery_subtask", "")
    key_column = discovery_entry["key_column"]
    columns = discovery_entry["columns"]
    _judge_url = judge_sglang_url or sglang_url

    # Parse GT for F1 evaluation.
    gt_answer = ground_truth.get("answer", "")
    if isinstance(gt_answer, pd.DataFrame):
        gt_df = gt_answer
    else:
        import io
        gt_df = pd.read_csv(io.StringIO(gt_answer), sep="|")
        gt_df = gt_df.loc[:, ~gt_df.columns.str.startswith("Unnamed")]
    gt_df.columns = [c.strip() for c in gt_df.columns]

    print(f"\n{'='*60}")
    print(f"Question: {question[:120]}...")
    print(f"Columns: {columns}, Key: {key_column}, GT rows: {len(gt_df)}")

    # === Phase 1: Discovery worker ===
    print("\n--- Phase 1: Discovery worker ---")
    print("  [Discovery] Sending subtask, waiting for worker...")
    discovered_cells = await run_one_cell_worker(
        session, sglang_url, search_client, discovery_subtask,
        tokenizer, toolcall_parser, tools_description_en,
        output_mode="keys", print_final=True,
    )
    discovered_entities = [c["row_key"] for c in discovered_cells if c["row_key"]]
    # Merge with GT to fill gaps.
    gt_entities = list(gt_df[key_column].astype(str).str.strip().unique())
    all_entities = list(dict.fromkeys(discovered_entities + gt_entities))
    print(f"  Discovered: {len(discovered_entities)}, GT: {len(gt_entities)} → merged: {len(all_entities)}")

    # === Phase 2: Template-based subtask generation (cell-level) ===
    non_key_cols = [c for c in columns if c != key_column]
    total_cells = len(all_entities) * len(non_key_cols)
    subtask_texts_by_cpw = {}

    for cpw in cells_per_worker_values:
        n_workers = math.ceil(total_cells / cpw)
        cells_per_worker = cpw

        # Group cells mechanically: one cell = one (entity, column) pair.
        # Each worker gets exactly cells_per_worker cells.
        subtasks = []
        cell_pool = []
        for entity in all_entities:
            for col in non_key_cols:
                cell_pool.append((entity, col))

        for i in range(0, len(cell_pool), cells_per_worker):
            batch = cell_pool[i:i + cells_per_worker]
            # Group by entity for cleaner text.
            by_entity = {}
            for entity, col in batch:
                if entity not in by_entity:
                    by_entity[entity] = []
                by_entity[entity].append(col)
            lines = []
            for entity, cols in by_entity.items():
                lines.append(
                    f"  - For the entry where {key_column} is \"{entity}\", "
                    f"find: {', '.join(cols)}"
                )
            subtasks.append(
                f"The main question is:\n{question}\n\n"
                f"Please search for and verify the following information:\n"
                + "\n".join(lines)
            )

        subtask_texts_by_cpw[str(cpw)] = subtasks
        print(f"  cpw={cpw} → {len(subtasks)} subtasks")

    # === Phase 3: Cell workers ===
    results = {}
    sem = asyncio.Semaphore(max_concurrent_workers)

    for cpw in cells_per_worker_values:
        subtask_texts = subtask_texts_by_cpw.get(str(cpw), [])
        if not subtask_texts:
            continue

        print(f"\n--- Phase 3: Cell workers (cpw={cpw}, n_workers={len(subtask_texts)}) ---")

        async def _bounded_worker(st):
            async with sem:
                return await run_one_cell_worker(
                    session, sglang_url, search_client, st,
                    tokenizer, toolcall_parser, tools_description_en,
                )

        tasks = [_bounded_worker(t) for t in subtask_texts]
        worker_results = await asyncio.gather(*tasks)

        # Collect cells.
        all_cells = []
        for wr in worker_results:
            all_cells.extend(wr)
        print(f"  Total cells written: {len(all_cells)}")

        # === Phase 4: Evaluate F1 ===
        table_state = {
            "key_column": key_column,
            "columns": columns,
            "rows": {},
        }
        for cell in all_cells:
            rk = cell["row_key"]
            if rk not in table_state["rows"]:
                table_state["rows"][rk] = {}
            table_state["rows"][rk][cell["column"]] = {"value": cell["value"]}

        eval_df = table_state_to_dataframe(table_state)
        if eval_df is None or eval_df.empty:
            f1 = 0.0
            print(f"  F1: {f1} (empty table)")
            results[cpw] = {"n_workers": len(subtask_texts), "f1_score": f1, "cells_written": len(all_cells)}
            continue

        pred_keys = set(eval_df[key_column].astype(str).tolist())
        gt_keys = set(gt_df[key_column].astype(str).tolist())
        matched = pred_keys & gt_keys
        print(f"  Pred rows: {len(eval_df)}, GT rows: {len(gt_df)}, Matched: {len(matched)}/{len(gt_keys)}")

        from rlinf.agents.wideseek_r1.utils.reward import evaluate_markdown

        async def judge_fn(messages):
            return await sglang_generate(session, _judge_url, messages)

        try:
            f1, fmt_ok = await evaluate_markdown(eval_df, ground_truth, judge_fn, norm_column_=False)
            print(f"  F1: {f1:.4f} (format_ok={fmt_ok})")
        except Exception as e:
            print(f"  F1 error: {e}")
            f1 = 0.0

        results[cpw] = {"n_workers": len(subtask_texts), "f1_score": f1, "cells_written": len(all_cells)}

    return results


def table_state_to_dataframe(table_state: dict) -> Optional[pd.DataFrame]:
    """Convert table_state to DataFrame (mirrors _table_state_to_dataframe)."""
    if not table_state or not table_state.get("rows"):
        return None
    columns = table_state.get("columns", [])
    key_col = table_state.get("key_column", columns[0] if columns else "key")
    if not columns:
        return None

    data = []
    for row_key, row_data in table_state["rows"].items():
        row = {col: None for col in columns}
        row[key_col] = row_key
        for col in columns:
            cell = row_data.get(col)
            if cell is not None and isinstance(cell, dict):
                row[col] = cell.get("value")
            elif cell is not None:
                row[col] = cell
        data.append(row)
    return pd.DataFrame(data, columns=columns)


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------
def plot_results(results: dict, output_path: str = "cell_oracle_results.png"):
    """Plot F1 vs N_workers."""
    items = sorted(results.items())
    cpw_values = [k for k, _ in items]
    n_workers = [v["n_workers"] for _, v in items]
    f1_scores = [v["f1_score"] for _, v in items]
    cells_written = [v["cells_written"] for _, v in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # F1 vs N_workers.
    ax1.plot(n_workers, f1_scores, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("N_workers (total)")
    ax1.set_ylabel("Item-level F1")
    ax1.set_title("F1 vs Parallelism")
    ax1.grid(True, alpha=0.3)
    for i, cpw in enumerate(cpw_values):
        ax1.annotate(
            f"cpw={cpw}",
            (n_workers[i], f1_scores[i]),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=8,
        )

    # Cells written vs N_workers.
    ax2.plot(n_workers, cells_written, "s-", linewidth=2, markersize=8, color="green")
    ax2.set_xlabel("N_workers (total)")
    ax2.set_ylabel("Cells written")
    ax2.set_title("Coverage vs Parallelism")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Cell-level Oracle F1 experiment")
    p.add_argument(
        "--dataset_path",
        required=True,
        help="Path to original JSONL dataset (for GT answers).",
    )
    p.add_argument(
        "--discovery_path",
        required=True,
        help="Path to discovery_subtasks JSONL (one subtask per line).",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="Number of samples to run (default: all).",
    )
    p.add_argument(
        "--sglang_url",
        default="http://127.0.0.1:30000",
        help="SGLang-compatible OpenAI endpoint for worker inference (policy model).",
    )
    p.add_argument(
        "--model_path",
        required=True,
        help="Path to model checkpoint (for loading tokenizer + chat template).",
    )
    p.add_argument(
        "--judge_sglang_url",
        default=None,
        help="SGLang endpoint for LLM judge (F1 evaluation). "
             "Defaults to --sglang_url if not set.",
    )
    p.add_argument(
        "--online",
        action="store_true",
        help="Use online Serper/Jina APIs instead of local RAG.",
    )
    p.add_argument(
        "--search_server",
        default="127.0.0.1:8000",
        help="Local RAG server address (when --online is not set).",
    )
    p.add_argument(
        "--cells_per_worker",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 10, 15, 20, 30],
        help="Cells-per-worker values to sweep (default: 1 2 4 6 10 15 20 30).",
    )
    p.add_argument(
        "--max_entities",
        type=int,
        default=50,
        help="Maximum entities to include in the experiment.",
    )
    p.add_argument(
        "--max_concurrent_workers",
        type=int,
        default=8,
        help="Max concurrent worker coroutines (limits parallel LLM calls).",
    )
    p.add_argument(
        "--output",
        default="cell_oracle_results.png",
        help="Output plot path.",
    )
    return p.parse_args()


async def main():
    import aiohttp

    args = parse_args()

    # Load original dataset (for GT).
    with open(args.dataset_path) as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]
    # Load discovery subtasks.
    with open(args.discovery_path) as f:
        discovery_entries = [json.loads(line) for line in f if line.strip()]

    # Match by index, subsample.
    paired = list(zip(raw_samples, discovery_entries))
    if args.num_samples < len(paired):
        import random
        random.seed(42)
        paired = random.sample(paired, args.num_samples)
    print(f"Loaded {len(paired)} paired samples")

    def _normalize(sample: dict) -> dict:
        q = sample.get("question") or sample.get("prompt")
        a = sample.get("answer", "")
        if isinstance(a, str):
            a = a.strip()
            if a.startswith("```markdown"): a = a[len("```markdown"):]
            if a.startswith("```"): a = a[3:]
            if a.endswith("```"): a = a[:-3]
            a = a.strip()
            lines = [l for l in a.split("\n") if not set(l.strip()).issubset(set("|-: ")) and "|" in l]
            a = "\n".join(lines)
        unique_cols = sample.get("unique_columns", [])
        return {
            "prompt": q,
            "answer": {"answer": a, "is_markdown": True,
                       "unique_columns": unique_cols,
                       "instance_id": sample.get("instance_id", sample.get("id", "")),
                       "language": sample.get("language", "en")},
        }

    # Load tokenizer and tool-call parser.
    from transformers import AutoTokenizer
    from rlinf.algorithms.toolcall_parsers import WideSeekQwenToolCallParser
    from rlinf.agents.wideseek_r1.utils.tool_description import tools_description_en

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    toolcall_parser = WideSeekQwenToolCallParser()
    search_client = _make_search_client(online=args.online, server_addr=args.search_server)

    async with aiohttp.ClientSession() as session:
        all_results = []
        for idx, (raw, disc) in enumerate(paired):
            question_data = _normalize(raw)
            pid = question_data["answer"].get("instance_id", idx)
            print(f"\n{'#'*60}")
            print(f"# Sample {idx+1}/{len(paired)}  (id={pid})")
            print(f"{'#'*60}")
            try:
                result = await run_experiment(
                    session=session,
                    sglang_url=args.sglang_url,
                    search_client=search_client,
                    question_data=question_data,
                    discovery_entry=disc,
                    cells_per_worker_values=args.cells_per_worker,
                    max_concurrent_workers=args.max_concurrent_workers,
                    judge_sglang_url=args.judge_sglang_url,
                    tokenizer=tokenizer,
                    toolcall_parser=toolcall_parser,
                    tools_description_en=tools_description_en,
                )
                result["_instance_id"] = pid
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR on sample {pid}: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate across samples.
    agg = {}
    for cpw in args.cells_per_worker:
        f1s = [r[cpw]["f1_score"] for r in all_results if cpw in r]
        nws = [r[cpw]["n_workers"] for r in all_results if cpw in r]
        cws = [r[cpw]["cells_written"] for r in all_results if cpw in r]
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        avg_nw = sum(nws) / len(nws) if nws else 0
        avg_cw = sum(cws) / len(cws) if cws else 0
        agg[cpw] = {
            "n_workers": round(avg_nw),
            "f1_score": avg_f1,
            "cells_written": round(avg_cw),
        }

    # Print summary.
    print("\n" + "=" * 60)
    print(f"SUMMARY  (averaged over {len(all_results)} samples)")
    print("-" * 60)
    for cpw, r in sorted(agg.items()):
        print(
            f"cells/worker={cpw:3d}  n_workers={r['n_workers']:3d}  "
            f"F1={r['f1_score']:.4f}  cells_written={r['cells_written']}"
        )

    # Plot aggregated.
    plot_results(agg, args.output)

    # Dump all data.
    dump_path = args.output.replace(".png", ".json")
    with open(dump_path, "w") as f:
        json.dump(
            {
                "aggregated": {str(k): v for k, v in agg.items()},
                "per_sample": [
                    {str(k): v for k, v in r.items() if not k.startswith("_")}
                    for r in all_results
                ],
            },
            f,
            indent=2,
        )
    print(f"Raw data saved to {dump_path}")


if __name__ == "__main__":
    asyncio.run(main())
