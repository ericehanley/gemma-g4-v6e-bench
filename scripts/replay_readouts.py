#!/usr/bin/env python3
"""Recompute the customer-replay readouts from raw loadgen steps.jsonl files.

Usage:
  python3 scripts/replay_readouts.py LABEL=path/to/steps.jsonl [LABEL=path ...] [--prefix conc_c] [--rate-prefix rate_]

For each label: closed-loop interpolations (req/s at p50 = 70 ms, at p99 = 110 ms), the highest
closed-loop point with p90 < 500 ms, the plateau, the Poisson rate at p50 = 70 ms (same logic as
pick_rate.py), and the soak summary. Several files may be given under one label (comma-separated)
when a run was split across pods (G4: results_run1 + results_run2); later files win on duplicate
labels, unless --keep-first is given.
"""
import argparse
import json


def load(paths, keep_first):
    steps = {}
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            s = json.loads(line)
            if keep_first and s["label"] in steps:
                continue
            steps[s["label"]] = s
    return list(steps.values())


def interp(points, metric, target):
    """points: list of (lat_dict, req_per_s, c); linear interpolation between the last point with
    metric <= target and the first with metric > target, ordered by concurrency."""
    pts = sorted(points, key=lambda p: p[2])
    under = [p for p in pts if p[0][metric] <= target]
    over = [p for p in pts if p[0][metric] > target]
    if not under or not over:
        return None
    (l0, r0, c0), (l1, r1, c1) = under[-1], over[0]
    f = (target - l0[metric]) / (l1[metric] - l0[metric])
    return r0 + f * (r1 - r0), c0 + f * (c1 - c0)


def poisson_rate_at(steps, prefix, target=70.0, metric="p50"):
    pts = []
    for s in steps:
        if s.get("mode") != "rate" or not s["label"].startswith(prefix) or not s.get("lat_ms"):
            continue
        offered, achieved = s["offered_rate"], s["req_per_s"] or 0.0
        kept_up = achieved >= 0.9 * offered and s["n_err"] <= 0.01 * max(1, s["n_ok"])
        pts.append((offered, s["lat_ms"][metric] if kept_up else float("inf")))
    if not pts:
        return None
    pts.sort()
    under = [p for p in pts if p[1] <= target]
    over = [p for p in pts if p[1] > target]
    if under and over:
        (r0, l0), (r1, l1) = under[-1], over[0]
        rate = r0 + (r1 - r0) * (0.5 if l1 == float("inf") else (target - l0) / (l1 - l0))
    elif under:
        rate = under[-1][0]
    else:
        rate = over[0][0]
    return max(pts[0][0], min(pts[-1][0], rate))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="LABEL=path[,path...]")
    ap.add_argument("--prefix", default="conc_c", help="closed-loop step label prefix")
    ap.add_argument("--rate-prefix", default="rate_", help="Poisson step label prefix")
    ap.add_argument("--keep-first", action="store_true", help="on duplicate labels keep the first file's step")
    args = ap.parse_args()
    for spec in args.inputs:
        label, paths = spec.split("=", 1)
        steps = load(paths.split(","), args.keep_first)
        conc = [(s["lat_ms"], s["req_per_s"], s["concurrency"]) for s in steps
                if s.get("mode") == "conc" and s["label"].startswith(args.prefix) and s.get("lat_ms")]
        print(f"== {label} ({len(conc)} closed-loop points, {len(steps)} steps)")
        for c, (lat, rps, _) in sorted((p[2], p) for p in conc):
            print(f"   c={c:<4d} {rps:7.1f} req/s  p50 {lat['p50']:7.1f}  p90 {lat['p90']:7.1f}  p99 {lat['p99']:7.1f}")
        r = interp(conc, "p50", 70.0)
        print(f"   closed loop @ p50=70 ms : {r[0]:.1f} req/s (c~{r[1]:.1f})" if r else "   closed loop @ p50=70 ms : n/a")
        r = interp(conc, "p99", 110.0)
        print(f"   closed loop @ p99=110 ms: {r[0]:.1f} req/s (c~{r[1]:.1f})" if r else "   closed loop @ p99=110 ms: n/a")
        ok = [p for p in conc if p[0]["p90"] < 500.0]
        if ok:
            best = max(ok, key=lambda p: p[1])
            print(f"   best point with p90<500  : {best[1]:.1f} req/s (c={best[2]}, p90 {best[0]['p90']:.0f})")
        if conc:
            top = max(conc, key=lambda p: p[1])
            print(f"   plateau (max req/s)      : {top[1]:.1f} req/s (c={top[2]})")
        pr = poisson_rate_at(steps, args.rate_prefix)
        print(f"   Poisson rate @ p50=70 ms : {pr:.0f} req/s" if pr is not None else "   Poisson rate @ p50=70 ms : n/a")
        pr99 = poisson_rate_at(steps, args.rate_prefix, 110.0, "p99")
        print(f"   Poisson rate @ p99=110 ms: {pr99:.0f} req/s" if pr99 is not None else "   Poisson rate @ p99=110 ms: n/a")
        for s in steps:
            if s["label"].startswith("soak_"):
                lat = s["lat_ms"]
                print(f"   soak {s['offered_rate']:.0f} req/s x {s['duration_s']:.0f} s: served {s['req_per_s']}, "
                      f"p50/p90/p99 {lat['p50']}/{lat['p90']}/{lat['p99']} ms, ok {s['n_ok']} err {s['n_err']}, "
                      f"hit {s['metrics_delta'].get('hit_rate')}, top1 {s['top1_is_label_rate']}, auc {s['p_unsafe']['auc_vs_label']}")
        errs = sum(s["n_err"] for s in steps)
        print(f"   total ok {sum(s['n_ok'] for s in steps)}, errors {errs}")


if __name__ == "__main__":
    main()
