# Gemma 4 E4B safety-classifier replay: one G4 GPU vs one TPU v6e chip

Replays a customer's real text-safety workload (safechat-100k: 100,000 chat conversations,
one-token classification with top-20 logprobs, score = sigmoid(logP(`Un`) − logP(`Safe`))) against
vLLM with the customer's exact `vllm serve` flags on **one RTX PRO 6000** (GKE `g4-standard-48`,
2026-10-08) and on **one TPU v6e chip** (GKE `ct6e-standard-4t`, server pinned to one chip,
2026-10-09), with the same dataset, prompt template, request shape, stand-in middleware, load
generator and step durations on both. The customer reported 190–240 req/s per GPU at
p50/p99 = 70/110 ms; the question was whether that replicates and what one v6e chip does instead.

Full write-ups: [G4 results](gemma-4-E4B/g4/customer_replay/RESULTS.md),
[v6e results and 1:1 comparison](gemma-4-E4B/v6e/customer_replay/RESULTS.md),
[independent QC of both](docs/qc_report_2026-10-09.md).

## Headline

| Readout | G4 (1 GPU) req/s | v6e (1 chip) req/s | G4 ÷ v6e | v6e ÷ G4 req/$ on-demand | req/$ spot |
| :--- | --: | --: | --: | --: | --: |
| Closed loop, p50 = 70 ms (customer SLO) | 192 | 132 | 1.45× | 1.15× | 1.85× |
| Closed loop, p99 = 110 ms (customer SLO) | 208 | 133 | 1.56× | 1.06× | 1.71× |
| Poisson arrivals, p50 = 70 ms | 175 | 125 | 1.40× | 1.19× | 1.91× |
| Max with p90 < 500 ms | 243 | 152 | 1.60× | 1.04× | 1.67× |
| Saturation | 249 | 201 | 1.24× | 1.35× | 2.16× |

Prices per accelerator-hour, us-central1 list: G4 $4.50 / $1.74 (on-demand / spot), v6e $2.70 / $0.65
(see [Price basis](#price-basis)). req/$ = req/s × 3600 / $/h.

* **Speed:** G4 serves 1.4–1.6× more req/s per accelerator at the customer's latency targets and
  meets the customer's own 190–240 req/s band (192 / 208); one v6e chip does not (132 / 133).
* **Cost:** per request, v6e is roughly at parity on-demand (1.04–1.19×) and 1.65–1.9× better on
  spot at us-central1 list — if a spot node can be kept (v6e spot in us-central1 was reclaimed three
  times in one afternoon during this work). The spot advantage is region-specific: at us-east5 spot
  prices (v6e $1.62 vs G4 $2.24 per accelerator-hour) it is 0.9–1.0× (1.1× at saturation).
* **Why the gap:** `--kv-sharing-fast-prefill` (+48–66 % on G4 for this model, depending on batch)
  is a no-op in vLLM-TPU (measured on/off at c ≤ 64; no implementation in tpu-inference). On the
  synthetic tie-back the two accelerators are close when neither uses it (82 vs 88 req/s at c = 64;
  v6e reaches 116 at c = 128 where G4 stays at 82). Without the flag G4 would land at an estimated
  115–130 req/s at p50 = 70 ms, i.e. about where the v6e chip is (estimate, not measured).
  Fast-prefill support in tpu-inference would be expected to put the v6e chip in the 190–220 req/s
  range.
* **TPU caveats for a deployment:** tpu-inference's default block size at 8192 context is 256
  tokens, which drops the prefix-cache hit rate to 0 % (−4 to −28 % throughput); `--block-size 32`
  restores parity. `allowed_token_ids` crashes the engine on the current nightly (not used in the v6e
  run; no effect on the score). TPU latency tails are heavier (p99/p50 4.8 vs 2.1 at the p50 = 70 ms
  operating point), the usable Poisson window is narrow (p50 62 → 136 → 644 ms at 125 / 140 / 160
  req/s), and cold start is ~22 min (18 min of XLA compilation) vs 4 min on G4.
* **Not measured:** 8-accelerator node scaling on either platform for this workload (the customer
  extrapolates 1 GPU × 8); G4 without fast-prefill on the real data.

## Price basis

Node list prices from the Cloud Billing Catalog API (service `6F81-5844-456A`), divided by the
number of accelerators per node. Two assumptions: the 1-accelerator shapes (`g4-standard-48`,
`ct6e-standard-1t`) are priced at ⅛ of the 8-accelerator nodes (the G4 SKUs scale linearly), and
both platforms are quoted at us-central1 although the v6e run physically happened in us-east5.

**g4-standard-384** (384 vCPU + 1,440 GiB + 8 GPU; the GPU SKU is named `RTX 6000 96GB`):

| Region | Mode | vCPU | RAM | GPU | Total | per GPU-h |
| :--- | :--- | --: | --: | --: | --: | --: |
| us-central1 | On-demand | $18.78 | $8.45 | $8.77 | $36.00/h | $4.50 |
| us-central1 | Spot | $6.41 | $2.88 | $4.66 | $13.94/h | $1.74 |
| us-east5 | On-demand | $20.66 | $9.30 | $9.64 | $39.60/h | $4.95 |
| us-east5 | Spot | $9.47 | $4.26 | $4.21 | $17.95/h | $2.24 |

**v6e-8** (`ct6e-standard-8t`):

| Region | Mode | Total | per chip-h |
| :--- | :--- | --: | --: |
| us-central1 | On-demand | $21.60/h | $2.70 |
| us-central1 | Spot | $5.19/h | $0.65 |
| us-east5 | Spot | $12.96/h | $1.62 |

The GPU is only 24 % of the G4 node bill; 52 % is the 384 vCPUs the shape forces you to buy. That
packaging, not the silicon, is what makes the per-dollar comparison close despite the 1.4–1.6× speed
advantage.

## Workload and method

* **Data:** safechat-100k, all 100,000 rows, streamed from GCS into the pod's RAM-backed volume and
  never written to disk. Rendered with the customer's `[s1]: text` line template through the Gemma
  chat template, thinking off: mean 352 prompt tokens (p50 350, p90 468, p99 685, max 4,042; 5.4 %
  over 512), 98-token shared instruction prefix → 27 % prefix-cache hit rate on both platforms.
  The plain `s1:` format of the customer's `content_str` column is 326 tokens and ~15–20 % cheaper.
  0 duplicate prompts, 0 consecutive rows sharing a context prefix.
* **Request:** `POST /v1/chat/completions`, 1 completion token, `top_logprobs = 20`,
  `allowed_token_ids = [39316, 2805]` (G4 only, see below), non-streaming, through a stand-in for
  the customer's middleware ([gdm_binary_probability.py](gemma-4-E4B/g4/customer_replay/scripts/gdm_binary_probability.py)).
* **Server:** `google/gemma-4-E4B-it` (same architecture as the customer's fine-tune), bf16,
  `--max-model-len 8192 --max-num-seqs 256 --max-num-batched-tokens 16384 --max-logprobs 20
  --enable-prefix-caching --kv-sharing-fast-prefill --gpu-memory-utilization 0.90 --seed 42`.
  G4: `vllm/vllm-openai:v0.30.0` with the customer's `--load-format=runai_streamer
  --attention-backend=TRITON_ATTN`. v6e: `vllm/vllm-tpu` nightly-20261006 (tpu-inference f5c194c).
* **Load** ([rbx_loadgen.py](gemma-4-E4B/g4/customer_replay/scripts/rbx_loadgen.py)): smoke →
  30 s warm-up → closed loop c = 1, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256 × 60 s (after a
  5 s warm-up each) → Poisson arrivals 2 min per rate → 10-min soak at the rate where p50 ≈ 70 ms →
  variants (file order, plain format, middleware bypass, no `allowed_token_ids`, …) → synthetic
  tie-back with `vllm bench serve` (50 + 450 random tokens). Non-repeating shuffled walk of the
  dataset; per-request records hold latency, status, prompt tokens, score and top-1 id — no text.
* **Readouts:** req/s at p50 = 70 ms and at p99 = 110 ms are linear interpolations between the two
  adjacent closed-loop points; "max with p90 < 500 ms" is the best measured point; the Poisson
  figure is the interpolated offered rate at p50 = 70 ms, confirmed by the 10-min soak.
  [scripts/replay_readouts.py](scripts/replay_readouts.py) regenerates all of them from
  `loadgen/steps.jsonl`.

## What differs on TPU

| | G4 | v6e | Why |
| :--- | :--- | :--- | :--- |
| `--load-format=runai_streamer`, `--attention-backend=TRITON_ATTN` | customer's | dropped | CUDA-only / load path only |
| `--block-size` | vLLM default (effective 32-token prefix-cache granularity) | `32` forced | TPU default at 8192 context is 256-token pages → 0 % hit rate; measured separately as `V1_bs256` |
| `--kv-sharing-fast-prefill` | +48–66 % | kept, measured no-op | no implementation in tpu-inference (`V1_nofp` variant) |
| `allowed_token_ids` | sent | not sent | crashes the tpu-inference sampler (`runner/input_batch.py:451`); score unaffected; measured nil on G4 |
| Poisson rate list | fixed 120–320 | 50–95 % of the measured plateau | platform-independent readout |

Details and reasons: [v6e README](gemma-4-E4B/v6e/customer_replay/README.md).

## Repository layout

```
README.md                          this file
docs/qc_report_2026-10-09.md       independent verification of every number in both reports
scripts/replay_readouts.py         headline numbers from loadgen/steps.jsonl
gemma-4-E4B/
  g4/customer_replay/              G4 replay: README, RESULTS, Job manifests, in-pod driver, load
                                   generator, middleware, dataset streamer
  g4/verify/                       fast-prefill A/B on one G4 GPU (synthetic spec) — only the summary
                                   and manifest are kept; cited by both results reports
  v6e/customer_replay/             v6e replay: README, RESULTS, manifests (1t pool / 4t host as run),
                                   driver, local launch/stream/watch helpers
```

Everything in `gemma-4-E4B/g4/customer_replay/scripts/` is shared: the v6e ConfigMap is built from
that directory plus `v6e/customer_replay/scripts/`.

## Running it

Each platform README has the exact sequence; in short:

1. `kubectl apply` the namespace/service-account, then the Job manifest
   ([rbx-replay.yaml](gemma-4-E4B/g4/customer_replay/rbx-replay.yaml) on G4;
   [local/launch.sh](gemma-4-E4B/v6e/customer_replay/local/launch.sh) on v6e builds the ConfigMap
   and applies in order).
2. From the laptop, stream the dataset into the pod's tmpfs with the operator's own credential
   ([stream_dataset.py](gemma-4-E4B/g4/customer_replay/stream_dataset.py) /
   [local/stream.sh](gemma-4-E4B/v6e/customer_replay/local/stream.sh)) and touch `/data/READY`.
3. The in-pod driver ([run_replay.sh](gemma-4-E4B/g4/customer_replay/scripts/run_replay.sh) /
   [run_replay_v6e.sh](gemma-4-E4B/v6e/customer_replay/scripts/run_replay_v6e.sh)) starts the
   server, gates on a smoke test, runs every step and writes `/workspace/output`; copy it out with
   `kubectl cp` (v6e: `local/copy_loop.sh` does this every 10 min as spot insurance).
4. Read out: `venv/bin/python scripts/replay_readouts.py "G4=…/results_run1_g4-48/loadgen/steps.jsonl,…/results_run2_g4-384/loadgen/steps.jsonl" --keep-first`
   and `… "v6e=…/results_run2_pm-mistral-vllm_gke-tpu-5315d3f8-5zhw/loadgen/steps.jsonl"`.

The only laptop-side Python dependency is `pyarrow` (for `stream_dataset.py`; it shells out to
`gcloud`/`kubectl`), kept in a local `venv/`; `replay_readouts.py` is stdlib only. Nothing is
installed globally.

## Raw data (not in git)

The `results_*/` directories (G4 85 MB, v6e 65 MB) and the flat CSV exports are kept locally by the
operator and are gitignored. Each run directory holds `loadgen/steps.jsonl` (one JSON per step),
`loadgen/samples/<step>.csv` (per request: start, latency, status, prompt tokens, score, top-1 id)
and `<step>_gauges.csv` (running/waiting/KV usage), `cpu.csv` (per-role CPU, 1 s), `gpu.csv`
(nvidia-smi, 1 s) or `tpu_info.log` (duty cycle, TensorCore %, HBM, 5 s), `server_*.log`
(start-up config dump + uvicorn access log), `server_args*.txt`, `metrics_*.prom`, `tieback/`,
`dataset_stats.json`, `progress.log`, `versions.txt`, `env.txt`. None of them contains prompt text.
The results reports and the QC report quote every number that matters from them.

## Data handling

The dataset is real user chat. It was streamed straight from GCS into pod RAM with the operator's
own credential, rendered there, and never written to any disk; result files hold only counts,
latencies and scores (verified by grep on both platforms, see the QC report); the benchmark
namespaces were deleted after each run. The repository contains the customer's prompt *template*
(in `prep_dataset.py`) but no conversation data. `gcs-key.json`, if present in the working copy,
is gitignored.

## Verification status

Every headline number in both results reports was independently recomputed from the per-request
samples, telemetry, server logs and Prometheus counters on 2026-10-09; all reproduce. The QC report
lists the three substantive corrections that were applied (region-specific spot claim, the unmeasured
no-fast-prefill estimate softened to "parity", TPU utilization telemetry found present) and the
remaining open items.
