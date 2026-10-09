#!/usr/bin/env python3
"""Pick the Poisson arrival rate whose p50 latency is closest to a target, from steps.jsonl.

Linear interpolation between the highest rate with p50 <= target and the lowest rate with
p50 > target (a rate the server did not keep up with counts as "over"). Clamped to the tested
range and rounded to a multiple of 5. Prints the rate only.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("--prefix", default="rate_")
    ap.add_argument("--target-p50-ms", type=float, default=70.0)
    ap.add_argument("--metric", default="p50", choices=["p50", "p90", "p99"])
    args = ap.parse_args()

    pts = []
    with open(args.steps) as f:
        for line in f:
            s = json.loads(line)
            if s.get("mode") != "rate" or not s["label"].startswith(args.prefix) or not s.get("lat_ms"):
                continue
            offered, achieved = s["offered_rate"], s["req_per_s"] or 0.0
            lat = s["lat_ms"][args.metric]
            kept_up = achieved >= 0.9 * offered and s["n_err"] <= 0.01 * max(1, s["n_ok"])
            pts.append((offered, lat if kept_up else float("inf")))
    if not pts:
        raise SystemExit("no rate steps found")
    pts.sort()
    under = [p for p in pts if p[1] <= args.target_p50_ms]
    over = [p for p in pts if p[1] > args.target_p50_ms]
    if under and over:
        (r0, l0), (r1, l1) = under[-1], over[0]
        if l1 == float("inf"):
            rate = r0 + (r1 - r0) * 0.5
        else:
            rate = r0 + (r1 - r0) * (args.target_p50_ms - l0) / (l1 - l0)
    elif under:
        rate = under[-1][0]
    else:
        rate = over[0][0]
    rate = max(pts[0][0], min(pts[-1][0], rate))
    print(int(5 * round(rate / 5.0)))


if __name__ == "__main__":
    main()
