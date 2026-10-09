#!/usr/bin/env python3
"""Diff two vLLM /metrics snapshots and print one JSON line.

Counters are summed across all label sets (engines, finish reasons, ...), so the result
is correct for any DP size. A metric that is missing from the snapshot is reported as null
rather than failing the benchmark step.
"""
import argparse
import json
import re

LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)")

# First name present wins; the alternates cover older vLLM spellings.
WANTED = {
    "queries": ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"],
    "hits": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
    "prompt_tokens": ["vllm:prompt_tokens_total"],
    "cached_prompt_tokens": ["vllm:prompt_tokens_cached_total"],
    "requests": ["vllm:request_success_total"],
}


def parse(path):
    sums = {}
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                m = LINE.match(line)
                if not m:
                    continue
                try:
                    value = float(m.group(2))
                except ValueError:
                    continue
                sums[m.group(1)] = sums.get(m.group(1), 0.0) + value
    except OSError:
        pass
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--num-prompts", type=int)
    args = ap.parse_args()

    before, after = parse(args.before), parse(args.after)
    out = {"label": args.label, "seed": args.seed, "num_prompts": args.num_prompts}
    for key, names in WANTED.items():
        out[key] = None
        for name in names:
            if name in after:
                out[key] = after[name] - before.get(name, 0.0)
                break

    q, h = out["queries"], out["hits"]
    out["hit_rate"] = round(h / q, 4) if q and h is not None else None
    r, p = out["requests"], out["prompt_tokens"]
    out["prompt_tokens_per_request"] = round(p / r, 1) if r and p is not None else None
    print(json.dumps(out))


if __name__ == "__main__":
    main()
