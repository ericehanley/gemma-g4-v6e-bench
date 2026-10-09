# Customer replay: safechat-100k on one G4 GPU with the customer's server config

Replays the customer's real dataset against vLLM v0.30.0 with their exact `vllm serve` flags on
one RTX PRO 6000, to check their reported 190–240 req/s at p50/p99 = 70/110 ms.

Result (2026-10-08): replicated. One GPU gives 192 req/s at p50 = 70 ms and 208 req/s at
p99 = 110 ms under fixed concurrency (~175 req/s at p50 = 70 ms under Poisson arrivals);
saturation 249 req/s, GPU at its 600 W cap. Their middleware and `allowed_token_ids` cost
nothing measurable. Full write-up and tables: [RESULTS.md](RESULTS.md).

## Layout

| Path | What |
| :--- | :--- |
| `rbx-replay.yaml` | Namespace `rbx-bench`, service account, Job pinned to `g4-48-spot-pool` (apply order in the header) |
| `stream_dataset.py` | Runs on the laptop: GCS parquet → memory → JSONL on stdout, piped into the pod's tmpfs. Nothing on disk |
| `scripts/run_replay.sh` | In-pod driver: customer flags verbatim, smoke gate, closed-loop sweep, Poisson steps, soak, variants, tie-back. `CONC_SWEEP` / `RATE_STEPS` env override for resuming after a preemption |
| `scripts/rbx_loadgen.py` | Load generator (closed loop / Poisson, non-repeating dataset cursor, `/metrics` deltas). Records no text |
| `scripts/gdm_binary_probability.py` | Stand-in for the customer's middleware (pure ASGI; honours `x-gdm-bypass: 1`) |
| `scripts/prep_dataset.py` | In-pod template rendering (both line formats) and aggregate length stats |
| `scripts/pick_rate.py` | Picks the soak rate (p50 ≈ 70 ms) from the Poisson steps |
| `scripts/cpumon.py`, `scripts/metrics_delta.py` | Telemetry helpers (per-role CPU sampler; `/metrics` counter deltas) |
| `results_run1_g4-48/` | Closed-loop sweep c=1…256 and Poisson 120 on the g4-standard-48 node (preempted afterwards). Local only, not in git |
| `results_run2_g4-384/` | Cross-check c=8/16/32, Poisson 160–320, 10-min soak at 175 req/s, variants, synthetic tie-back, on one GPU of the g4-standard-384 node. Local only, not in git |

Each results dir: `loadgen/steps.jsonl` (one JSON object per step), `loadgen/samples/*.csv`
(per-request latency / prompt tokens / score, no text), `gpu.csv` (1 s), `cpu.csv` (per role),
`server_*.log`, `dataset_stats.json`, `progress.log`.

## Data handling

The dataset is real user chat. It was streamed straight from GCS into the pod's RAM-backed
`/data` volume with the operator's own credential, rendered there, and never written to disk
anywhere. Result files hold only counts, latencies and scores (checked: no template text in any
result file). The `rbx-bench` namespace was deleted at the end of the run.
