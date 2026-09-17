#!/usr/bin/env python3
"""Convert vLLM `bench serve` result JSONs into the benchmark summary CSV.

Produces exactly the column order used by the existing v6e result CSVs so that
G4 and v6e runs can be diffed or concatenated directly.

Usage:
    python3 scripts/results_to_csv.py <results_dir> <output_csv>
"""

import csv
import json
import os
import re
import sys

# Column order is fixed to match the existing *_results.csv files in this repo.
COLUMNS = [
    "max_concurrency", "num_prompts", "completed", "failed", "duration",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_e2el_ms", "median_e2el_ms", "p50_e2el_ms", "p90_e2el_ms",
    "p95_e2el_ms", "p99_e2el_ms", "std_e2el_ms",
    "mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p90_ttft_ms",
    "p95_ttft_ms", "p99_ttft_ms", "std_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p50_tpot_ms", "p90_tpot_ms",
    "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p50_itl_ms", "p90_itl_ms",
    "p95_itl_ms", "p99_itl_ms",
    "total_input_tokens", "total_output_tokens", "max_concurrent_requests",
    "max_output_tokens_per_s", "model_id", "tokenizer_id", "date", "backend",
    "endpoint_type", "label", "request_rate", "burstiness", "request_goodput",
    "rtfx", "std_tpot_ms", "std_itl_ms",
]


def concurrency_of(filename: str) -> int:
    """Sort key: native_c128_results.json -> 128."""
    match = re.search(r"_c(\d+)_", filename)
    return int(match.group(1)) if match else 0


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    results_dir, output_csv = sys.argv[1], sys.argv[2]

    files = sorted(
        (f for f in os.listdir(results_dir) if f.endswith(".json")),
        key=concurrency_of,
    )
    if not files:
        print(f"No JSON results found in {results_dir}")
        return 1

    rows = []
    for name in files:
        with open(os.path.join(results_dir, name)) as handle:
            data = json.load(handle)
        # `latencies` and `queue_times` are per-request lists; drop them.
        rows.append({col: data.get(col, "") for col in COLUMNS})

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
