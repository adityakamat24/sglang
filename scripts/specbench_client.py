"""
Simple SpecBench-style driver for SGLang suffix decoding benchmarks.

Usage:
    python sglang/scripts/specbench_client.py \
        --dataset question.jsonl \
        --server http://127.0.0.1:21000 \
        --max-new-tokens 256 \
        --concurrencies 1 4 16 64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import requests


def _find_first_text(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        preferred_keys = ("prompt", "question", "instruction", "text", "content")
        for key in preferred_keys:
            if key in value:
                res = _find_first_text(value[key])
                if res:
                    return res
        for v in value.values():
            res = _find_first_text(v)
            if res:
                return res
    if isinstance(value, list):
        for elem in value:
            res = _find_first_text(elem)
            if res:
                return res
    return None


def extract_prompt(item: dict) -> str:
    """Try to extract the first user prompt from the record."""
    for key in ("prompt", "question", "instruction"):
        if key in item:
            res = _find_first_text(item[key])
            if res:
                return res
    turns = item.get("turns")
    res = _find_first_text(turns)
    if res:
        return res
    if "input" in item:
        res = _find_first_text({"instruction": item.get("instruction", ""), "input": item["input"]})
        if res:
            return res
    raise KeyError(f"Unable to find prompt text in record keys: {list(item.keys())}")


def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [extract_prompt(json.loads(line)) for line in f]


def fetch_server_metrics(server: str) -> dict[str, Any]:
    def _fetch(path: str) -> dict[str, Any]:
        resp = requests.get(f"{server}/{path}", timeout=10)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()

    info = _fetch("get_server_info")
    metrics = _fetch("get_metrics")
    return {"info": info, "metrics": metrics}


def reset_server_metrics(server: str):
    try:
        requests.get(f"{server}/flush_cache", timeout=10)
        requests.get(f"{server}/clear_metrics", timeout=10)
    except requests.RequestException:
        pass


def send_request(server: str, prompt: str, max_new_tokens: int) -> tuple[float, int]:
    payload = {
        "text": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
    }
    start = time.time()
    resp = requests.post(f"{server}/generate", json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Request failed with status {resp.status_code}: {resp.text[:200]}"
        )
    end = time.time()
    result = resp.json()
    text_entries = result.get("text") or result.get("outputs") or []
    if isinstance(text_entries, list):
        text = text_entries[0] if text_entries else ""
    elif isinstance(text_entries, str):
        text = text_entries
    else:
        text = ""
    if not text:
        out_ids = result.get("output_ids")
        if isinstance(out_ids, list) and out_ids:
            text = str(out_ids[0])
    token_count = len(text.split()) if text else len(result.get("output_ids", []))
    return end - start, token_count


@dataclass
class RunMetrics:
    method: str
    spec_len: int
    concurrency: int
    throughput_tok_s: float
    avg_latency_s: float
    time_per_token_ms: float
    drafted_tokens: int
    accepted_tokens: int
    avg_spec_accept_length: float


def run_concurrency(
    server: str,
    prompts: List[str],
    max_new_tokens: int,
    concurrency: int,
) -> tuple[float, float, int]:
    total_tokens = 0
    latencies = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, server, prompt, max_new_tokens)
            for prompt in prompts
        ]
        for future in as_completed(futures):
            dt, tokens = future.result()
            latencies.append(dt)
            total_tokens += tokens
    elapsed = time.time() - start
    throughput = total_tokens / elapsed
    avg_latency = sum(latencies) / len(latencies)
    return throughput, avg_latency, len(prompts)


def save_table(results: List[RunMetrics], path: str):
    headers = [
        "method",
        "spec_len",
        "concurrency",
        "time_per_token_ms",
        "throughput_tok_s",
        "drafted_tokens",
        "accepted_tokens",
        "avg_spec_accept_length",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                r.method,
                str(r.spec_len),
                str(r.concurrency),
                f"{r.time_per_token_ms:.4f}",
                f"{r.throughput_tok_s:.4f}",
                str(r.drafted_tokens),
                str(r.accepted_tokens),
                f"{r.avg_spec_accept_length:.4f}",
            ]
            f.write(",".join(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="SpecBench-style client for SGLang.")
    parser.add_argument("--dataset", required=True, help="Path to question.jsonl")
    parser.add_argument("--server", default="http://127.0.0.1:21000", help="Server URL")
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Generation length"
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--method",
        default="suffix",
        help="Label to include in the summary (e.g., suffix_cache_on)",
    )
    parser.add_argument(
        "--spec-len",
        type=int,
        help="Spec length label (defaults to --max-new-tokens)",
    )
    parser.add_argument(
        "--output-table",
        default="benchmark_results.csv",
        help="CSV file to write the summary table",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.dataset)
    print(
        f"Loaded {len(prompts)} prompts from {args.dataset}. "
        f"Testing server {args.server}."
    )
    results: List[RunMetrics] = []
    spec_len_label = args.spec_len if args.spec_len is not None else args.max_new_tokens
    reset_server_metrics(args.server)
    for conc in args.concurrencies:
        throughput, avg_latency, num_prompts = run_concurrency(
            args.server, prompts, args.max_new_tokens, conc
        )
        time_per_token_ms = 1000.0 / throughput if throughput > 0 else math.inf
        metrics = fetch_server_metrics(args.server)
        internal = metrics["info"].get("internal_states", [{}])[0]
        drafted = int(metrics["metrics"].get("spec_decode_num_draft_tokens_total", 0)) if metrics["metrics"] else 0
        accepted = int(metrics["metrics"].get("spec_decode_num_accepted_tokens_total", 0)) if metrics["metrics"] else 0
        avg_accept = float(internal.get("avg_spec_accept_length", 0.0))
        metrics = RunMetrics(
            method=args.method,
            spec_len=spec_len_label,
            concurrency=conc,
            throughput_tok_s=throughput,
            avg_latency_s=avg_latency,
            time_per_token_ms=time_per_token_ms,
            drafted_tokens=drafted,
            accepted_tokens=accepted,
            avg_spec_accept_length=avg_accept,
        )
        results.append(metrics)
        print(
            f"[{args.method}] conc={conc}: tp={throughput:.2f} tok/s "
            f"(tpot={time_per_token_ms:.2f} ms), drafted={drafted}, accepted={accepted}, "
            f"avg_accept_len={avg_accept:.2f}"
        )
        reset_server_metrics(args.server)

    print("\nSummary:")
    headers = [
        "method",
        "spec_len",
        "concurrency",
        "time_per_token_ms",
        "throughput_tok_s",
        "drafted_tokens",
        "accepted_tokens",
        "avg_spec_accept_length",
    ]
    print("\t".join(headers))
    for r in results:
        row = [
            r.method,
            str(r.spec_len),
            str(r.concurrency),
            f"{r.time_per_token_ms:.2f}",
            f"{r.throughput_tok_s:.2f}",
            str(r.drafted_tokens),
            str(r.accepted_tokens),
            f"{r.avg_spec_accept_length:.2f}",
        ]
        print("\t".join(row))
    save_table(results, args.output_table)
    print(f"\nSaved CSV to {args.output_table}")


if __name__ == "__main__":
    main()

