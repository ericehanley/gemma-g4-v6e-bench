# Customer replay: safechat-100k on one TPU v6e chip with the customer's server config

The 1:1 counterpart of [`../../g4/customer_replay/`](../../g4/customer_replay/README.md): the same
real dataset, template, one-token logprob request, stand-in middleware, load generator and step
durations, replayed against vLLM-TPU on **one v6e chip** (one chip of an on-demand
`ct6e-standard-4t`, server pinned to chip 0; the `rbx-replay-v6e.yaml` manifest targets a
`ct6e-standard-1t`, 44 vCPU, 176 GB) instead of one RTX PRO 6000 (`g4-standard-48`, 48 vCPU, 180 GB).

G4 reference (2026-10-08, one GPU): 192 req/s at p50 = 70 ms, 208 at p99 = 110 ms (closed loop),
~175 req/s at p50 = 70 ms under Poisson arrivals, saturation 249 req/s.

**Result (2026-10-09, one v6e chip):** 132 req/s at p50 = 70 ms, 133 at p99 = 110 ms (closed loop),
~125 req/s at p50 = 70 ms under Poisson arrivals, saturation 201 req/s; 0 errors in 451k requests,
27.2 % prefix-cache hit rate (same as G4). Per accelerator G4 is 1.4–1.6× faster at the latency
targets; per dollar (us-central1 list; the spot gap closes to ~1× at us-east5 spot prices) v6e is
1.04–1.19× better on-demand and 1.65–1.9× on spot.
`--kv-sharing-fast-prefill` is a no-op on TPU (measured); the TPU default block size (256) gives a
0 % hit rate and costs 4–28 % (14–28 % at c ≤ 32). Full write-up: [RESULTS.md](RESULTS.md).
The run happened on the idle on-demand `ct6e-standard-4t` node of cluster `pm-mistral-vllm`
(us-east5-b), server pinned to one chip (`rbx-replay-v6e-mistral-4t.yaml`), after three spot
preemptions in us-central1.

## What differs from the G4 run, and why

| | G4 | v6e | Reason |
| :--- | :--- | :--- | :--- |
| Image | `vllm/vllm-openai:v0.30.0` | `vllm/vllm-tpu` nightly-20261006 (tpu-inference f5c194c / vLLM 72e7874), by digest | TPU stack is JAX-based |
| `--load-format=runai_streamer`, `--attention-backend=TRITON_ATTN` | customer's | dropped | not available / CUDA-only; load path and kernel selection only |
| `--block-size` | vLLM default (CacheConfig 16; effective prefix-cache granularity 32 — "kv lcm block sizes 32" for this hybrid model) | **`32` forced** (fallback 64 → 128 → default) | tpu-inference's default at `--max-model-len 8192` is 256-token pages, which would cache none of the 98-token shared prefix (G4 measured a 27 % hit rate). The TPU default is measured separately (`V1_bs256`) |
| `--kv-sharing-fast-prefill` | customer's (+66 % on G4) | kept, **expected no-op** | tpu-inference has no fast-prefill implementation; measured on and off (`V1_nofp`) |
| `allowed_token_ids` | sent (client + middleware) | **not sent** (`ALLOWED_TOKEN_IDS=0`: client `--no-allowed`, middleware strips it via `GDM_NO_ALLOWED_TOKEN_IDS=1`) | tpu-inference's sampler path crashes the engine (`runner/input_batch.py:451` TypeError, on `main` as of 2026-10-09). The score only needs logP(Un) and logP(Safe) from the top-20; top-1 was Safe/Un on 100 % of requests; nil cost on G4 |
| Poisson rates | fixed list 120–320 | 50–95 % of the measured closed-loop plateau | same readout (interpolated rate at p50 = 70 ms), platform-independent rate list |
| Telemetry | `nvidia-smi` 1 s (util %, W, SM MHz) | `tpu-info` 5 s (duty cycle, TensorCore %, HBM) | comparable "accelerator busy" metrics: G4 util 91–100 %, v6e duty 90–100 % at the operating points |

Everything else (data path, `rbx_loadgen.py`, `gdm_binary_probability.py`, `prep_dataset.py`,
`pick_rate.py`, `cpumon.py`, `metrics_delta.py`) is taken **unchanged** from the G4 directory; the
ConfigMap is built from both directories.

## Layout

| Path | What |
| :--- | :--- |
| `rbx-replay-v6e.yaml` | Namespace `rbx-bench`, service account, Job for a dedicated `ct6e-standard-1t` pool (pool name selectable in the manifest) |
| `rbx-replay-v6e-mistral-4t.yaml` | **As run:** Job for an idle on-demand `ct6e-standard-4t` node (all 4 chips requested, server pinned to chip 0 via `TPU_VISIBLE_CHIPS` / process bounds) |
| `scripts/run_replay_v6e.sh` | In-pod driver: V1 (customer config: smoke → closed loop c = 1…256 → Poisson → 10-min soak → variants → synthetic tie-back), then `V1_nofp` and `V1_bs256` on their own servers. Resumable via env (`VARIANTS`, `CONC_SWEEP`, `RATE_STEPS`, `SOAK_RATE_OVERRIDE`, …) |
| `local/launch.sh` | Ordered apply (namespace/SA → secret → ConfigMap → Job) |
| `local/stream.sh` | Laptop → pod tmpfs dataset stream with identity check and row-count gate |
| `local/watch.sh`, `local/copy_loop.sh` | Milestone watcher; 10-min copy-out loop (spot insurance) |
| `results_run*_<node>/` | Raw outputs: `progress.log`, `loadgen/steps.jsonl`, `loadgen/samples/`, `tpu_info.log`, `cpu.csv`, `server_*.log`, `dataset_stats.json`, `tieback/`. Local only, not in git |

## Data handling

Same as the G4 run: dedicated namespace, data streamed with the operator's corp credential straight
into the pod's RAM-backed `/data`, never on any disk; result files hold counts, latencies and
scores only; the namespace is deleted at the end.
