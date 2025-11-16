#!/usr/bin/env python3
"""
Parse benchmark results and generate formatted tables similar to vLLM PR #25784.

Usage:
    python3 parse_results.py [--results-dir RESULTS_DIR]
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_csv_file(filepath: str) -> List[Dict[str, str]]:
    """Parse a CSV file and return list of row dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def aggregate_results(results_dir: Path) -> Tuple[Dict, Dict]:
    """
    Aggregate results from all CSV files.

    Returns:
        (specbench_results, blazedit_results) where each is a nested dict:
        results[method][spec_len][concurrency] = {metric: value}
    """
    specbench_results = defaultdict(lambda: defaultdict(dict))
    blazedit_results = defaultdict(lambda: defaultdict(dict))

    for csv_file in results_dir.glob("*.csv"):
        filename = csv_file.name

        # Determine benchmark type and method
        if filename.startswith("specbench_"):
            results_dict = specbench_results
        elif filename.startswith("blazedit_"):
            results_dict = blazedit_results
        else:
            continue

        # Parse the results
        rows = parse_csv_file(csv_file)
        for row in rows:
            method = row['method']
            spec_len = int(row['spec_len'])
            concurrency = int(row['concurrency'])

            if concurrency not in results_dict[method][spec_len]:
                results_dict[method][spec_len][concurrency] = {}

            results_dict[method][spec_len][concurrency].update({
                'time_per_token_ms': float(row['time_per_token_ms']),
                'throughput': float(row.get('throughput_tok_s', 0)),
                'drafted_tokens': int(row.get('drafted_tokens', 0)),
                'accepted_tokens': int(row.get('accepted_tokens', 0)),
                'avg_spec_accept_length': float(row.get('avg_spec_accept_length', 0)),
            })

    return specbench_results, blazedit_results


def format_table(results: Dict, metric_name: str, metric_key: str,
                concurrencies: List[int], spec_lens: List[int],
                methods: List[str]) -> str:
    """Format results into a markdown table."""

    # Header
    header = "| method | spec_len | " + " | ".join(f"concurrency {c}" for c in concurrencies) + " |"
    separator = "|--------|----------|" + "|".join("----------" for _ in concurrencies) + "|"

    lines = [f"\n### {metric_name}\n", header, separator]

    # Data rows
    for method in methods:
        for spec_len in spec_lens:
            row_data = [method, str(spec_len)]
            for conc in concurrencies:
                value = results.get(method, {}).get(spec_len, {}).get(conc, {}).get(metric_key, 0)
                if metric_key == 'time_per_token_ms':
                    row_data.append(f"{value:.2f}")
                elif metric_key in ['drafted_tokens', 'accepted_tokens']:
                    row_data.append(f"{int(value)}")
                else:
                    row_data.append(f"{value:.2f}")

            lines.append("| " + " | ".join(row_data) + " |")

    return "\n".join(lines)


def generate_report(specbench_results: Dict, blazedit_results: Dict,
                   output_file: str = None):
    """Generate a comprehensive markdown report."""

    concurrencies = [1, 4, 16, 64]
    spec_lens = [5, 12, 32]

    # Determine available methods
    all_methods = set()
    for results in [specbench_results, blazedit_results]:
        all_methods.update(results.keys())

    # Sort methods for consistent ordering
    methods_order = [
        "suffix (w/ cache)",
        "suffix (w/o cache)",
        "ngram [5, 5]",
        "ngram [3, 5]",
    ]
    methods = [m for m in methods_order if m in all_methods]

    report_lines = [
        "# SGLang Suffix Decoding Benchmark Results",
        "",
        "This report contains benchmark results comparing suffix decoding with ngram speculation.",
        "",
        "Benchmarks were run on Specbench and Blazedit datasets with varying concurrency levels.",
        "",
    ]

    # Specbench results
    if specbench_results:
        report_lines.append("## Specbench")
        report_lines.append("")

        tables = [
            ("Time per output token (ms)", "time_per_token_ms"),
            ("Total drafted tokens", "drafted_tokens"),
            ("Total accepted tokens", "accepted_tokens"),
            ("Average acceptance length", "avg_spec_accept_length"),
        ]

        for table_name, metric_key in tables:
            table = format_table(
                specbench_results,
                table_name,
                metric_key,
                concurrencies,
                spec_lens,
                methods
            )
            report_lines.append(table)
            report_lines.append("")

    # Blazedit results
    if blazedit_results:
        report_lines.append("## Blazedit")
        report_lines.append("")

        tables = [
            ("Time per output token (ms)", "time_per_token_ms"),
            ("Total drafted tokens", "drafted_tokens"),
            ("Total accepted tokens", "accepted_tokens"),
            ("Average acceptance length", "avg_spec_accept_length"),
        ]

        for table_name, metric_key in tables:
            table = format_table(
                blazedit_results,
                table_name,
                metric_key,
                concurrencies,
                spec_lens,
                methods
            )
            report_lines.append(table)
            report_lines.append("")

    # Summary statistics
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("### Speedup Analysis")
    report_lines.append("")

    # Calculate average speedups
    for benchmark_name, results in [("Specbench", specbench_results), ("Blazedit", blazedit_results)]:
        if not results:
            continue

        report_lines.append(f"#### {benchmark_name}")
        report_lines.append("")

        # Compare suffix w/ cache vs ngram [5,5]
        suffix_cache = results.get("suffix (w/ cache)", {})
        ngram_5_5 = results.get("ngram [5, 5]", {})

        if suffix_cache and ngram_5_5:
            speedups = []
            for spec_len in spec_lens:
                for conc in concurrencies:
                    suffix_tpot = suffix_cache.get(spec_len, {}).get(conc, {}).get('time_per_token_ms', 0)
                    ngram_tpot = ngram_5_5.get(spec_len, {}).get(conc, {}).get('time_per_token_ms', 0)

                    if suffix_tpot > 0 and ngram_tpot > 0:
                        speedup = ngram_tpot / suffix_tpot
                        speedups.append(speedup)

            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                report_lines.append(
                    f"- Average speedup (suffix w/ cache vs ngram [5,5]): **{avg_speedup:.2f}x**"
                )

        report_lines.append("")

    report = "\n".join(report_lines)

    # Output
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")
    else:
        print(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Parse and format suffix decoding benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing benchmark result CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="RESULTS.md",
        help="Output markdown file for the report",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1

    print(f"Parsing results from: {results_dir}")
    specbench_results, blazedit_results = aggregate_results(results_dir)

    if not specbench_results and not blazedit_results:
        print("ERROR: No results found to parse")
        return 1

    print(f"Found results for:")
    if specbench_results:
        print(f"  - Specbench: {len(specbench_results)} methods")
    if blazedit_results:
        print(f"  - Blazedit: {len(blazedit_results)} methods")

    output_file = Path(args.results_dir) / args.output
    generate_report(specbench_results, blazedit_results, str(output_file))

    return 0


if __name__ == "__main__":
    exit(main())
