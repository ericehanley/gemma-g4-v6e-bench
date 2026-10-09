# Customer replay on one TPU v6e chip — results and 1:1 comparison with one G4 GPU

*Repository copy of the session report, with the corrections from the [QC report](../../../docs/qc_report_2026-10-09.md) of 2026-10-09 applied. The `results_run*/` directories referenced below are kept locally by the operator and are not in git.*

*Measured 2026-10-09 18:00–19:29 UTC on `gke-tpu-5315d3f8-5zhw` (cluster `pm-mistral-vllm`, us-east5-b, on-demand `ct6e-standard-4t`, server pinned to chip 0). Same dataset, template, request shape, stand-in middleware, load generator and step durations as the G4 replay of 2026-10-08 ([customer_replay_results.md](../../g4/customer_replay/RESULTS.md)). Raw data: [results_run2_pm-mistral-vllm_gke-tpu-5315d3f8-5zhw/](results_run2_pm-mistral-vllm_gke-tpu-5315d3f8-5zhw).*

## 1. Headline: one RTX PRO 6000 vs one v6e chip, customer workload

Prices are us-central1 list, per accelerator-hour: G4 $4.50 on-demand / $1.74 spot (`g4-standard-48`), v6e $2.70 / $0.65 (`ct6e-standard-1t`). req/$ = req/s × 3600 / $/h.

| Readout | G4 req/s | v6e req/s | G4 ÷ v6e (speed) | v6e ÷ G4 req/$, on-demand | req/$, spot |
| :--- | --: | --: | --: | --: | --: |
| Closed loop, **p50 = 70 ms** | 192 | **132** | 1.45× | **1.15×** | **1.85×** |
| Closed loop, **p99 = 110 ms** | 208 | **133** | 1.56× | 1.06× | 1.71× |
| Poisson arrivals, p50 = 70 ms | 175 | 125 | 1.40× | 1.19× | 1.91× |
| Max with p90 < 500 ms | 243 | 152 | 1.60× | 1.04× | 1.67× |
| Saturation | 249 | 201 | 1.24× | 1.35× | 2.16× |

* Per accelerator, **G4 serves 1.4–1.6× more requests per second at the customer's latency targets**; the v6e chip is cheaper, so **per dollar v6e is roughly at parity on-demand (1.04–1.19×) and 1.65–1.9× better on spot at us-central1 list prices** — the spot gap is region-specific: at the us-east5 spot prices in the root README's price basis (v6e $1.62, G4 $2.24 per accelerator-hour) it is 0.9–1.0× (1.1× at saturation). See §6.
* The customer's own band, 190–240 req/s per accelerator at p50/p99 = 70/110 ms, is met by the G4 GPU (192 / 208) and **not by one v6e chip (132 / 133)**.
* Sanity on v6e: 100 % of answers are `Safe`/`Un`, both ids in the top-20 on 99.9 %, prefix-cache hit rate **27.2 %** (G4: 27.3 %), prompt tokens 352 (G4: 352), AUC vs. labels 0.61–0.64 (G4: 0.61–0.64), **0 errors in 451,421 requests** (all steps incl. warm-ups and variants).

## 2. Closed loop (60 s per point, measured after 5 s warm-up)

| c | v6e req/s | v6e p50 / p90 / p99 ms | G4 req/s | G4 p50 / p90 / p99 ms |
| --: | --: | :-- | --: | :-- |
| 1 | 55.3 | 17 / 19 / 26 | 63.6 | 16 / 17 / 23 |
| 4 | 107.0 | 37 / 38 / 52 | 114.2 | 35 / 40 / 45 |
| 8 | 132.2 | 59 / 68 / 88 | 146.2 | 54 / 61 / 72 |
| 12 | 132.8 | 98 / 108 / 114 | 185.6 | 64 / 73 / 86 |
| 16 | 139.8 | 115 / 146 / 149 | 201.3 | 79 / 89 / 97 |
| 20 | 140.6 | 150 / 151 / 158 | 211.3 | 94 / 104 / 117 |
| 24 | 146.0 | 153 / 223 / 226 | 216.6 | 111 / 122 / 137 |
| 32 | 139.0 | 231 / 232 / 235 | 225.1 | 141 / 155 / 181 |
| 48 | 151.6 | 314 / 316 / 460 | 234.0 | 204 / 224 / 245 |
| 64 | 131.2 | 513 / 515 / 519 | 241.6 | 262 / 288 / 309 |
| 96 | 172.8 | 515 / 641 / 668 | 243.2 | 393 / 422 / 454 |
| 128 | 191.9 | 659 / 682 / 723 | 247.5 | 516 / 541 / 568 |
| 256 | 201.5 | 1270 / 1279 / 1576 | 249.3 | 1024 / 1030 / 1282 |

Interpolated v6e: p50 = 70 ms → 132.4 req/s (c ≈ 9); p99 = 110 ms → 132.7 req/s (c ≈ 11); p90 < 500 ms → 151.6 (c = 48 — a single 60 s window; c = 32 and c = 64 gave 139 and 131); plateau 201.5 at c = 256.

The v6e curve has a distinctive shape: throughput is flat at ~130–150 req/s from c = 8 to c = 64 while latency grows linearly with c, then rises to ~200 only at c ≥ 96. Host CPU is not the cause (engine core 0.1–0.3 cores, API server 0.2–0.45 cores, load generator 0.03 cores, host 0.3 % busy). The chip is: `tpu_info` shows duty cycle 90–100 % and TensorCore utilization 58–66 % from c = 8 upward (27 % at c = 1), flat at 64–66 % from c = 32 to c = 256 while throughput still rises 139 → 201 — i.e. a large share of chip time at mid-batch goes to padding and non-MXU work (sampling, the top-20 logprob gather over the 262k vocabulary). The dip at c = 64 is deterministic: 64 × ~257 computed tokens exceeds `--max-num-batched-tokens 16384`, the batch splits 63 + 1 and the padded 1-request step costs ~200 ms per round (per-10 s throughput is flat at 129–134 req/s; 76 % of latencies in one 25 ms bucket). The same split produces the Poisson cliff between 140 and 160 req/s (§3). See §5.

## 3. Open loop, Poisson arrivals (2 min per rate)

| Offered req/s | Served | v6e p50 / p90 / p99 ms | G4 at the same offered rate |
| --: | --: | :-- | :-- |
| 100 | 99.8 | 41 / 71 / 121 | — (120: 41 / 61 / 80) |
| 120 | 119.8 | 56 / 110 / 192 | 41 / 61 / 80 |
| **125 (10-min soak)** | **124.8** | **62 / 142 / 299** — 74,860 req, 0 errors | (175 soak: 70 / 104 / 144) |
| 140 | 139.8 | 136 / 309 / 589 | — |
| 160 | 159.9 | 644 / 891 / 957 | 60 / 89 / 119 |
| 180 | 180.1 | 808 / 953 / 1028 | — |
| 190 | 190.4 | 862 / 994 / 1097 | 83 / 130 / 185 |

Under random arrivals the v6e chip holds p50 = 70 ms at ~125 req/s (G4: ~175) and p99 = 110 ms below the lowest tested rate (p99 = 121 ms at 100 req/s; G4: ~150). Its tail is markedly heavier (p99/p50 = 4.8 at the soak vs. 2.1 on G4): TPU steps are coarse (60–500 ms per batch), so a request arriving just after a step starts waits for the whole step.

## 4. Variants

| Variant (v6e) | Result | Read |
| :--- | :--- | :--- |
| **`--kv-sharing-fast-prefill` removed** | c = 8/16/32/64: 129 / 140 / 139 / 137 req/s vs. 132 / 140 / 139 / 131 with the flag; c = 128: 166 vs. 192 (the two servers settled into different batch-split states, see §2; both windows internally flat); Poisson 125: p50 64 vs. 62 ms, p99 551 vs. 299 (2-min vs 10-min window) | **No-op on TPU at c ≤ 64, as predicted from the source** (tpu-inference has no fast-prefill implementation). On G4 the same flag is worth +66 % |
| **TPU default block size (256 tokens)** | hit rate **0.0 %**; c = 8/16/32/64/128: 109 / 121 / 100 / 125 / 142 req/s (−18 / −14 / −28 / −4 / −26 %; the −4 % at c = 64 is against the depressed V1 point); c = 8 p50 72 vs. 59 ms | A customer running these flags unmodified on TPU would lose the prefix cache entirely; `--block-size 32` is required for cache parity |
| Plain `s1:` line format (326 tokens) | p50 50 ms vs. 62 at 125 req/s; hit rate 29.5 % | ~20 % cheaper (G4: ~15 %) |
| Middleware bypassed | 62.8 vs. 62.2 ms p50 at 125 req/s; c = 16: 140 vs. 150 req/s | Middleware cost nil (same as G4) |

## 5. Tie-back to the earlier synthetic measurements

`vllm bench serve`, 50 + 450 random tokens, 1 output token, same server:

| c | v6e 1 chip req/s | p50 / p99 ms |
| --: | --: | :-- |
| 8 | 106.2 | 74 / 107 |
| 16 | 85.8 | 186 / 215 |
| 64 | 88.1 | 719 / 864 |
| 128 | 115.7 | 1192 / 1489 |

Consistent with the earlier 8-chip v6e sweeps divided by eight (635 → 79 and 805 → 101 per chip) and with G4's **no-fast-prefill** single-GPU figure of 82 req/s (its fast-prefill figure was 136). Everything measured before and after therefore lines up: on this spec the two chips are close when neither uses fast-prefill, and the G4 advantage on the real workload is the fast-prefill feature.

> [!IMPORTANT]
> **The structural gap is software, not silicon.** Gemma 4 E4B shares KV for 18 of its 42 layers; vLLM-CUDA's `--kv-sharing-fast-prefill` skips those layers for all but the last prompt token, which on G4 lifted throughput ×1.66 on the synthetic spec. tpu-inference ignores the flag and runs every prompt token through all 42 layers. Scaling G4's real-workload number by the synthetic fast-prefill ratio gives ~115–130 req/s at p50 = 70 ms without the flag (1.66× at saturation, 1.48–1.54× at c = 8–16, which is where p50 = 70 ms sits; shorter real prompts dilute the saving further) — i.e. **roughly at parity with the v6e chip's 132** (an estimate, not a measurement). If fast-prefill lands in tpu-inference, the v6e chip should move into the 190–220 req/s range on this workload.

## 6. Deviations from the G4 run and other caveats

| Item | Detail |
| :--- | :--- |
| `allowed_token_ids` not sent | tpu-inference's sampler path for it crashes the engine on the first request (`runner/input_batch.py:451`, `jnp.zeros(max_num_reqs, vocab_size, dtype=…)` → `TypeError: got multiple values for argument 'dtype'`; still present on upstream `main` as of today — the feature has never worked on TPU). Client sends `--no-allowed`, the stand-in middleware strips it. The score uses logP(Un) − logP(Safe) from the top-20, which the mask does not change; top-1 was Safe/Un on 100 % of requests anyway; on G4 the flag measured nil |
| `--block-size 32` added | TPU default at 8192 context is 256-token pages → 0 % hit rate (measured, §4). Both measured |
| `--load-format=runai_streamer`, `--attention-backend=TRITON_ATTN` dropped | CUDA-only / load path only (the TPU image does register its own `runai_streamer` loader; irrelevant to serving speed) |
| `--kv-sharing-fast-prefill` | kept; no-op (§4) |
| Stack | `vllm/vllm-tpu` nightly-20261006 (tpu-inference f5c194c, vLLM core 0.30.1rc1.dev427, JAX 0.11.0, libtpu 0.0.44) vs. `vllm/vllm-openai:v0.30.0` on G4. KV cache bf16 on both (`kv_cache_dtype=auto`) |
| Host | ct6e-standard-4t (180 vCPU, 720 GB) with the server pinned to one chip (`TPU_VISIBLE_CHIPS=0`, `TPU_PROCESS_BOUNDS=1,1,1`, `TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1`; verified `total_devices=1`), not a ct6e-standard-1t (44 vCPU). CPU is idle either way (§2), so the per-chip result transfers |
| Location / price | Run in us-east5-b; prices quoted at us-central1 list for both platforms (root README, price basis). Spot req/$ assumes you can keep a spot node: today v6e spot in us-central1 was reclaimed three times within 14–30 min across all three zones (§7) |
| Telemetry | `tpu_info` 5 s: duty cycle, TensorCore utilization and HBM (30.3–31.0 / 31.24 GiB during steps; chips 1–3 unattached throughout). CPU per role in `cpu.csv` (API server = `serve_main`) |
| Model | public `google/gemma-4-E4B-it`, not the customer's fine-tune (same as G4) |
| Start-up | 22 min from `vllm serve` to healthy (1302 s; 18 min of it XLA compilation of the 16384-token / 256-seq / top-20-logprob buckets; ~23 min from pod start); 16 min for the block-size variant, 9 min for the no-fast-prefill variant (compile cache). G4: 3.75 min |
| Server warnings | `server_V1_bs32.log` contains 30,006 `tpu_runner.py:1634 'Should not schedule a request that does nothing!'` warnings (5,282 / 1,395 in the two variants), concentrated at c ≤ 4, c = 16–64 and the soak; no error, no visible throughput effect; to be raised upstream |

## 7. What it took (infrastructure notes for the record)

* us-central1 was unusable for a single-chip run today: a new `ct6e-standard-1t` **spot** pool was preempted at 30, 14 and 19 minutes (the 15:00Z reclaim took every v6e spot VM in the project across a/b/c); **flex-start** cycled a → b → a for 40 min without a grant; **on-demand** failed with `Invalid Reservation` (reservation-only in this project); the colleague-owned 1t nodes (kroukoz, pm-panw-jina, gccpiler) were all busy. Final run: the idle on-demand 4t node in `pm-mistral-vllm` (us-east5), per your "use an idle cluster" instruction. Everything created today has been deleted (namespaces on both clusters, the three 1t pools).
* Harness lessons captured in [README.md](README.md): GKE Warden requires `cloud.google.com/gke-tpu-topology` on TPU pods; `--tpu-topology` on a multi-zone pool is rejected; `tokeninfo` cannot introspect bound tokens (identity check now uses the active account + a bucket read).

## 8. Files

| Path | What |
| :--- | :--- |
| [gemma-4-E4B/v6e/customer_replay/](README.md) | Manifests (`rbx-replay-v6e.yaml` for a 1t pool, `rbx-replay-v6e-mistral-4t.yaml` as run), `scripts/run_replay_v6e.sh`, `local/` helpers, README with the deviation table |
| `results_run2_pm-mistral-vllm_gke-tpu-5315d3f8-5zhw/` | `progress.log`, `loadgen/steps.jsonl` (one JSON per step), `loadgen/samples/*.csv` (per request: latency, status, prompt tokens, score — no text), `cpu.csv`, `tpu_info.log`, `server_V1_bs32.log`, `server_V1_nofp_bs32.log`, `server_V1_bs256_bsdefault.log`, `tieback/`, `dataset_stats.json`, `versions.txt` |
| `results_run1_gke-tpu-e40e49ca-szln/` | First attempt (us-central1-b spot): start-up logs only (copied before the smoke test, so the `allowed_token_ids` traceback is not on disk — it is quoted in §6 from the live log); no dataset stats; the node was preempted after the smoke test |
| `customer_replay_v6e_closed_loop_by_concurrency.csv`, `customer_replay_v6e_open_loop_by_rate.csv` | Flat exports (gitignored like the G4 CSVs) |
| `gemma-4-E4B/g4/customer_replay/scripts/gdm_binary_probability.py` | One backward-compatible change: `GDM_NO_ALLOWED_TOKEN_IDS=1` strips `allowed_token_ids` |
