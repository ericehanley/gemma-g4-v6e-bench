#!/usr/bin/env python3
"""Load generator for the safechat-100k replay against a vLLM server.

Why not `vllm bench serve`: the customer does not use it, their requests are one-token
classifications with logprobs, and we need the dataset walked in a non-repeating order.

Modes
  conc   closed loop: --concurrency workers, each sends the next prompt when its last one
         returns. Measured over --duration seconds after --warmup seconds.
  rate   open loop: Poisson arrivals at --rate req/s for --duration seconds, each request in
         its own task (bounded by --max-inflight). Then a --drain window to collect the tail.

Dataset order: a seeded permutation of the prompt list walked with a cursor persisted in
--cursor, so consecutive steps never resend a prompt until the whole set is used (then it
reshuffles with seed+1). --order file replays rows in file order instead.

Request (default): POST /v1/chat/completions, one user turn with the customer's template,
max 1 completion token, temperature 0, top-20 logprobs as token ids, allowed_token_ids =
[Safe, Un] (drop with --no-allowed), thinking off, non-streaming. --endpoint completions sends
the chat-formatted text to /v1/completions instead. --bypass-mw adds `x-gdm-bypass: 1`, which
the stand-in middleware honours by passing the request through untouched.

Output (nothing from the prompts is written): one JSON line per step in <outdir>/steps.jsonl,
per-request samples in <outdir>/samples/<label>.csv (start, latency, status, prompt tokens,
p_unsafe, top-1 id), server gauges sampled from /metrics, and the /metrics counter deltas.
"""
import argparse
import asyncio
import json
import math
import os
import random
import re
import statistics
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gdm_binary_probability as mw  # noqa: E402  (shared response parsing)

SAFE_ID, UN_ID = mw.SAFE_ID, mw.UN_ID
# Gemma chat wrapper for one user turn, used only for the /v1/completions variant. vLLM adds BOS.
CHAT_PREFIX = "<start_of_turn>user\n"
CHAT_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"

PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)")
COUNTERS = {
    "queries": ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"],
    "hits": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
    "prompt_tokens": ["vllm:prompt_tokens_total"],
    "cached_prompt_tokens": ["vllm:prompt_tokens_cached_total"],
    "requests": ["vllm:request_success_total"],
    "preemptions": ["vllm:num_preemptions_total"],
}
GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
}


# ----------------------------------------------------------------------------- helpers
def parse_prom(text):
    sums = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = PROM_LINE.match(line)
        if not m:
            continue
        try:
            sums[m.group(1)] = sums.get(m.group(1), 0.0) + float(m.group(2))
        except ValueError:
            pass
    return sums


def counter_delta(before, after):
    out = {}
    for key, names in COUNTERS.items():
        out[key] = None
        for name in names:
            if name in after:
                out[key] = after[name] - before.get(name, 0.0)
                break
    q, h = out["queries"], out["hits"]
    out["hit_rate"] = round(h / q, 4) if q and h is not None else None
    r, p = out["requests"], out["prompt_tokens"]
    out["prompt_tokens_per_request"] = round(p / r, 1) if r and p is not None else None
    return out


def pct(sorted_xs, p):
    if not sorted_xs:
        return None
    k = (len(sorted_xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_xs[int(k)]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def dist(xs, nd=1):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), nd + 1),
        "p50": round(pct(xs, 50), nd),
        "p90": round(pct(xs, 90), nd),
        "p95": round(pct(xs, 95), nd),
        "p99": round(pct(xs, 99), nd),
        "max": round(xs[-1], nd),
    }


def auc(scores, labels):
    """Rank AUC of p_unsafe against is_violative (sanity check of the pipeline, not a metric)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None and l is not None]
    pos = sum(1 for _, l in pairs if l)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda x: x[0])
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return round((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg), 4)


class Cursor:
    """Walks a seeded permutation of range(n); state persists across invocations."""

    def __init__(self, n, path, order, seed):
        self.n, self.path, self.order = n, path, order
        self.state = {}
        if os.path.exists(path):
            with open(path) as f:
                self.state = json.load(f)
        st = self.state.get(order, {"seed": seed, "pos": 0})
        self.seed, self.pos = int(st["seed"]), int(st["pos"])
        self.pos_start, self.seed_start, self.wraps = self.pos, self.seed, 0
        self._perm()

    def _perm(self):
        self.perm = list(range(self.n))
        if self.order == "shuffle":
            random.Random(self.seed).shuffle(self.perm)

    def next(self):
        if self.pos >= self.n:
            self.seed += 1
            self.pos = 0
            self.wraps += 1
            self._perm()
        i = self.perm[self.pos]
        self.pos += 1
        return i

    def save(self):
        self.state[self.order] = {"seed": self.seed, "pos": self.pos}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    def summary(self):
        return {"order": self.order, "seed_start": self.seed_start, "pos_start": self.pos_start,
                "seed_end": self.seed, "pos_end": self.pos, "wraps": self.wraps}


# ----------------------------------------------------------------------------- requests
def build_body(args, prompt):
    if args.endpoint == "chat":
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": args.top_logprobs,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
    else:
        body = {
            "model": args.model,
            "prompt": CHAT_PREFIX + prompt + CHAT_SUFFIX,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": args.top_logprobs,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
            "stream": False,
        }
    if args.allowed:
        body["allowed_token_ids"] = [SAFE_ID, UN_ID]
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def parse_response(data, is_chat):
    """-> (p_unsafe, both_in_topk, top1_id, prompt_tokens) or raises."""
    resp = json.loads(data)
    if "p_unsafe" in resp:  # middleware format
        usage = resp.get("usage") or {}
        return resp.get("p_unsafe"), bool(resp.get("both_in_topk")), resp.get("top1_token_id"), usage.get("prompt_tokens")
    top1, entries = mw._first_token_logprobs(resp, is_chat)
    lu, ls = entries.get(UN_ID), entries.get(SAFE_ID)
    if lu is not None and ls is not None:
        p = 1.0 / (1.0 + math.exp(-(lu - ls)))
    elif lu is not None:
        p = 1.0
    elif ls is not None:
        p = 0.0
    else:
        p = None
    usage = resp.get("usage") or {}
    return p, lu is not None and ls is not None, top1, usage.get("prompt_tokens")


class Run:
    def __init__(self, args, prompts, labels, cursor):
        self.args, self.prompts, self.labels, self.cursor = args, prompts, labels, cursor
        self.url = args.base_url.rstrip("/") + ("/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions")
        self.headers = {"Content-Type": "application/json"}
        if args.bypass_mw:
            self.headers["x-gdm-bypass"] = "1"
        self.records = []  # (t_start, t_end, lat_ms, status, ptoks, p, both, top1, idx)
        self.gauge_samples = []
        self.t0 = None
        self.sent = 0

    async def one(self, session, idx):
        prompt = self.prompts[idx]
        payload = build_body(self.args, prompt)
        t_start = time.perf_counter()
        status, ptoks, p, both, top1 = 0, None, None, False, None
        try:
            async with session.post(self.url, data=payload, headers=self.headers) as resp:
                data = await resp.read()
                status = resp.status
                if status == 200:
                    try:
                        p, both, top1, ptoks = parse_response(data, self.args.endpoint == "chat")
                    except (ValueError, KeyError, IndexError, TypeError):
                        status = -2  # 200 but unparseable
        except asyncio.TimeoutError:
            status = -1
        except aiohttp.ClientError:
            status = -3
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001  - never let one request kill the run
            status = -4
        t_end = time.perf_counter()
        self.records.append((t_start - self.t0, t_end - self.t0, (t_end - t_start) * 1000.0, status, ptoks, p, both, top1, idx))

    async def gauge_sampler(self, session):
        url = self.args.base_url.rstrip("/") + "/metrics"
        while True:
            await asyncio.sleep(self.args.metrics_interval)
            try:
                async with session.get(url) as resp:
                    sums = parse_prom(await resp.text())
                self.gauge_samples.append((time.perf_counter() - self.t0,) + tuple(sums.get(GAUGES[k]) for k in ("running", "waiting", "kv_usage")))
            except Exception:  # noqa: BLE001
                pass

    async def run_conc(self, session):
        a = self.args
        self.t0 = time.perf_counter()
        t_measure = self.t0 + a.warmup
        t_end = t_measure + a.duration

        async def worker():
            while True:
                if time.perf_counter() >= t_end:
                    return
                if a.max_requests and self.sent >= a.max_requests:
                    return
                self.sent += 1
                await self.one(session, self.cursor.next())

        await asyncio.gather(*[worker() for _ in range(a.concurrency)])
        return {"t_measure": t_measure - self.t0, "t_end": t_end - self.t0, "sent": self.sent, "dropped": 0, "max_inflight": a.concurrency}

    async def run_rate(self, session):
        a = self.args
        rng = random.Random(a.seed * 7919 + 17)
        self.t0 = time.perf_counter()
        t_end = self.t0 + a.duration
        next_t = self.t0
        inflight = set()
        dropped, max_inflight = 0, 0
        while True:
            next_t += rng.expovariate(a.rate)
            if next_t >= t_end:
                break
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if len(inflight) >= a.max_inflight:
                dropped += 1
                continue
            task = asyncio.create_task(self.one(session, self.cursor.next()))
            inflight.add(task)
            task.add_done_callback(inflight.discard)
            self.sent += 1
            if len(inflight) > max_inflight:
                max_inflight = len(inflight)
            if a.max_requests and self.sent >= a.max_requests:
                break
        if inflight:
            _, pending = await asyncio.wait(inflight, timeout=a.drain)
            for t in pending:
                t.cancel()
        return {"t_measure": 0.0, "t_end": t_end - self.t0, "sent": self.sent, "dropped": dropped, "max_inflight": max_inflight}

    def summarize(self, window, counters_before, counters_after, wall_start):
        a = self.args
        if a.mode == "conc":
            # Steady state: count requests that completed inside the measurement window.
            recs = [r for r in self.records if window["t_measure"] <= r[1] <= window["t_end"]]
            if a.max_requests:  # smoke: whole run
                recs = list(self.records)
            if recs:
                span = (max(r[1] for r in recs) - min(r[0] for r in recs)) if a.max_requests else a.duration
            else:
                span = a.duration
        else:
            # Open loop: every request sent in the window counts, however late it finished.
            recs = list(self.records)
            span = a.duration
        ok = [r for r in recs if r[3] == 200]
        err = [r for r in recs if r[3] != 200]
        errors = {}
        for r in err:
            errors[str(r[3])] = errors.get(str(r[3]), 0) + 1
        lat = [r[2] for r in ok]
        ptoks = [r[4] for r in ok if r[4] is not None]
        ps = [r[5] for r in ok]
        p_vals = [p for p in ps if p is not None]
        labels = [self.labels[r[8]] if self.labels else None for r in ok]
        gauges = self.gauge_samples
        def gsum(i, nd=2):
            vals = [g[i] for g in gauges if g[i] is not None]
            return {"mean": round(statistics.fmean(vals), nd), "max": round(max(vals), nd)} if vals else None
        out = {
            "label": a.label,
            "mode": a.mode,
            "concurrency": a.concurrency if a.mode == "conc" else None,
            "offered_rate": a.rate if a.mode == "rate" else None,
            "endpoint": a.endpoint,
            "order": a.order,
            "allowed_token_ids": a.allowed,
            "bypass_mw": a.bypass_mw,
            "top_logprobs": a.top_logprobs,
            "duration_s": a.duration,
            "warmup_s": a.warmup,
            "wall_start": wall_start,
            "n_ok": len(ok),
            "n_err": len(err),
            "errors": errors,
            "req_per_s": round(len(ok) / span, 2) if span > 0 else None,
            "completed_in_window_per_s": round(sum(1 for r in ok if r[1] <= window["t_end"]) / span, 2) if span > 0 else None,
            "lat_ms": dist(lat),
            "prompt_tokens": dist(ptoks, 0),
            "p_unsafe": {
                "n": len(p_vals),
                "n_null": len(ps) - len(p_vals),
                "mean": round(statistics.fmean(p_vals), 4) if p_vals else None,
                "frac_gt_0_5": round(sum(1 for p in p_vals if p > 0.5) / len(p_vals), 4) if p_vals else None,
                "auc_vs_label": auc(ps, labels) if self.labels else None,
            },
            "both_in_topk_rate": round(sum(1 for r in ok if r[6]) / len(ok), 4) if ok else None,
            "top1_is_label_rate": round(sum(1 for r in ok if r[7] in (SAFE_ID, UN_ID)) / len(ok), 4) if ok else None,
            "client": {"sent": window["sent"], "dropped": window["dropped"], "max_inflight": window["max_inflight"],
                       "records_total": len(self.records)},
            "server_gauges": {"running": gsum(1), "waiting": gsum(2), "kv_usage": gsum(3, 4)},
            "metrics_delta": counter_delta(counters_before, counters_after),
            "cursor": self.cursor.summary(),
        }
        return out

    def write_samples(self, path):
        with open(path, "w") as f:
            f.write("t_start_s,t_end_s,lat_ms,status,prompt_tokens,p_unsafe,top1_id\n")
            for r in self.records:
                p = "" if r[5] is None else f"{r[5]:.4f}"
                f.write(f"{r[0]:.3f},{r[1]:.3f},{r[2]:.2f},{r[3]},{'' if r[4] is None else r[4]},{p},{'' if r[7] is None else r[7]}\n")

    def write_gauges(self, path):
        with open(path, "w") as f:
            f.write("t_s,running,waiting,kv_usage\n")
            for g in self.gauge_samples:
                f.write(",".join("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in g) + "\n")


async def fetch_metrics(session, base_url):
    try:
        async with session.get(base_url.rstrip("/") + "/metrics") as resp:
            return parse_prom(await resp.text())
    except Exception:  # noqa: BLE001
        return {}


async def main_async(args):
    with open(args.prompts) as f:
        prompts = json.load(f)
    labels = None
    if args.labels and os.path.exists(args.labels):
        with open(args.labels) as f:
            labels = json.load(f)
        if len(labels) != len(prompts):
            labels = None
    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)
    cursor = Cursor(len(prompts), args.cursor or os.path.join(args.outdir, "cursor.json"), args.order, args.seed)
    run = Run(args, prompts, labels, cursor)

    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=120, force_close=False)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    wall_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        before = await fetch_metrics(session, args.base_url)
        run.t0 = time.perf_counter()
        sampler = asyncio.create_task(run.gauge_sampler(session))
        try:
            window = await (run.run_conc(session) if args.mode == "conc" else run.run_rate(session))
        finally:
            sampler.cancel()
        after = await fetch_metrics(session, args.base_url)
    cursor.save()
    summary = run.summarize(window, before, after, wall_start)
    with open(os.path.join(args.outdir, "steps.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    run.write_samples(os.path.join(args.outdir, "samples", f"{args.label}.csv"))
    run.write_gauges(os.path.join(args.outdir, "samples", f"{args.label}_gauges.csv"))
    lat = summary["lat_ms"] or {}
    md = summary["metrics_delta"]
    pt = summary["prompt_tokens"] or {}
    print(
        f"{args.label}: {summary['req_per_s']} req/s, p50 {lat.get('p50')} / p90 {lat.get('p90')} / "
        f"p99 {lat.get('p99')} ms, ok {summary['n_ok']} err {summary['n_err']} {summary['errors'] or ''}, "
        f"prompt tok mean {pt.get('mean')}, hit rate {md.get('hit_rate')}, "
        f"running {((summary['server_gauges']['running'] or {}).get('mean'))}, "
        f"waiting {((summary['server_gauges']['waiting'] or {}).get('mean'))}, "
        f"both_in_topk {summary['both_in_topk_rate']}, auc {summary['p_unsafe']['auc_vs_label']}",
        flush=True,
    )
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSON array of prompt strings")
    ap.add_argument("--labels", default=None, help="JSON array of bools (is_violative), same order")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=["conc", "rate"], required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=None, help="conc mode: seconds before measuring (default 5)")
    ap.add_argument("--drain", type=float, default=30.0, help="rate mode: seconds to wait for in-flight requests")
    ap.add_argument("--max-requests", type=int, default=0, help="stop after this many sends (smoke test)")
    ap.add_argument("--max-inflight", type=int, default=4096)
    ap.add_argument("--order", choices=["shuffle", "file"], default="shuffle")
    ap.add_argument("--cursor", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--no-allowed", dest="allowed", action="store_false", help="omit allowed_token_ids")
    ap.add_argument("--bypass-mw", action="store_true", help="send x-gdm-bypass: 1")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--metrics-interval", type=float, default=5.0)
    args = ap.parse_args()
    if args.warmup is None:
        args.warmup = 5.0 if args.mode == "conc" else 0.0
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
