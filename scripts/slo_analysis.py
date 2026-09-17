#!/usr/bin/env python3
"""Summarise benchmark CSVs against the Roblox p90 E2E latency SLOs.

The workload is binary classification: 500 input tokens (50 fixed prefix),
1 output token. The goals are to maximise throughput subject to
p90 E2E < 1000 ms, and to locate the 500 ms p90 knee.

Usage:
    python3 scripts/slo_analysis.py <label>=<csv> [<label>=<csv> ...]
"""

import csv
import sys

SLOS = (500.0, 1000.0)

# Runs shorter than this are burst measurements, not steady state. The client
# uses num_prompts=1000 below concurrency 96, which at ~900 req/s drains in
# ~1.1s -- the server never leaves its empty-queue transient, so throughput is
# inflated and the number is not reproducible. Exclude these from "best" picks.
MIN_TRUSTWORTHY_DURATION_S = 5.0


def load(path):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if not row.get("max_concurrency"):
            continue
        out.append({
            "concurrency": int(float(row["max_concurrency"])),
            "rps": float(row["request_throughput"]),
            "p90": float(row["p90_e2el_ms"]),
            "p99": float(row["p99_e2el_ms"]),
            "mean": float(row["mean_e2el_ms"]),
            "duration": float(row["duration"]),
        })
    return sorted(out, key=lambda r: r["concurrency"])


def best_under(rows, slo):
    """Highest throughput among points meeting the p90 SLO.

    Only considers runs long enough to represent steady state.
    """
    ok = [r for r in rows
          if r["p90"] < slo and r["duration"] >= MIN_TRUSTWORTHY_DURATION_S]
    return max(ok, key=lambda r: r["rps"]) if ok else None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    runs = []
    for arg in sys.argv[1:]:
        label, _, path = arg.partition("=")
        runs.append((label, load(path)))

    for slo in SLOS:
        print(f"\n=== Max sustained throughput with p90 E2E < {slo:.0f} ms ===")
        print(f"  (steady-state only: runs shorter than "
              f"{MIN_TRUSTWORTHY_DURATION_S:.0f}s excluded)")
        print(f"{'run':<28}{'conc':>6}{'req/s':>10}{'p90 ms':>10}{'p99 ms':>10}{'dur s':>8}")
        for label, rows in runs:
            best = best_under(rows, slo)
            if best is None:
                print(f"{label:<28}{'—':>6}{'never met':>10}")
                continue
            print(f"{label:<28}{best['concurrency']:>6}{best['rps']:>10.1f}"
                  f"{best['p90']:>10.1f}{best['p99']:>10.1f}{best['duration']:>8.1f}")

    print(f"\n=== Peak throughput regardless of latency ===")
    print(f"{'run':<28}{'conc':>6}{'req/s':>10}{'p90 ms':>10}{'dur s':>8}  note")
    for label, rows in runs:
        peak = max(rows, key=lambda r: r["rps"])
        note = ("BURST - not steady state"
                if peak["duration"] < MIN_TRUSTWORTHY_DURATION_S else "")
        print(f"{label:<28}{peak['concurrency']:>6}{peak['rps']:>10.1f}"
              f"{peak['p90']:>10.1f}{peak['duration']:>8.1f}  {note}")

    print("\n=== Full p90 curves (req/s @ p90 ms) ===")
    for label, rows in runs:
        print(f"\n{label}:")
        for r in rows:
            flag = ""
            if r["duration"] < MIN_TRUSTWORTHY_DURATION_S:
                flag = f"  [burst {r['duration']:.1f}s - unreliable]"
            elif r["p90"] >= 1000:
                flag = "  <-- over 1000ms"
            elif r["p90"] >= 500:
                flag = "  <-- over 500ms"
            print(f"   c={r['concurrency']:<5} {r['rps']:>8.1f} req/s  "
                  f"p90={r['p90']:>9.1f} ms{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
