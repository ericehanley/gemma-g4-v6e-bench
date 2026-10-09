# Customer Replay Results: safechat-100k on One G4 GPU, Customer's Exact Server Config

*Measured 2026-10-08. Repository copy of the session report, with the corrections from the [QC report](../../../docs/qc_report_2026-10-09.md) of 2026-10-09 applied. The `results_run*/` directories referenced below are kept locally by the operator and are not in git.*

> [!IMPORTANT]
> **Replicated.** With the customer's exact vLLM 0.30.0 command line, their real dataset and one RTX PRO 6000, one GPU delivers **192 req/s at p50 = 70 ms** and **208 req/s at p99 = 110 ms** under fixed concurrency, or **~175 req/s at p50 = 70 ms** under random (Poisson) arrivals. The customer reported **190–240 req/s at p50/p99 = 70/110 ms**. We land at the low end of their band, inside the ±15% criterion, and their stack costs them nothing measurable: the GPU is the limit (91–100% busy, pinned at its 600 W cap near saturation). This is hypothesis H1.

## 1. What ran

| | Detail |
| :--- | :--- |
| Machine | Run 1: `g4-standard-48`, 1× RTX PRO 6000 (spot). Preempted after the closed-loop sweep. Run 2: one GPU of the `g4-standard-384` node (same GPU; cross-checked, see §2) |
| Image | `vllm/vllm-openai:v0.30.0` (digest `8a69ffad…`); torch 2.13.0+cu130, triton 3.7.1, transformers 5.17.0 |
| Server | Their command line verbatim: `--tensor-parallel-size=1 --load-format=runai_streamer --attention-backend=TRITON_ATTN --seed=42 --gpu-memory-utilization=0.90 --max-model-len=8192 --max-num-seqs=256 --max-num-batched-tokens=16384 --kv-sharing-fast-prefill --max-logprobs=20 --dtype=bfloat16 --enable-prefix-caching --middleware=gdm_binary_probability.GdmBinaryProbabilityMiddleware`; env `OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Model | `google/gemma-4-E4B-it` (their 0922 checkpoint is a fine-tune of the same architecture, so same speed) |
| Middleware | Stand-in ([gdm_binary_probability.py](scripts/gdm_binary_probability.py)): buffers request and response in the API server, fills in the one-token settings, rewrites the response to `p_unsafe = sigmoid(logP(Un) − logP(Safe))` |
| Data | safechat-100k, all 100,000 conversations, their template (`[s1]: text` lines), one user turn through the chat template, thinking off. Streamed GCS → pod RAM; nothing on disk; namespace deleted afterwards |
| Request | `/v1/chat/completions`, 1 completion token, `top_logprobs=20`, `allowed_token_ids=[39316, 2805]`, non-streaming |
| Load | [rbx_loadgen.py](scripts/rbx_loadgen.py): closed loop (c workers) and open loop (Poisson), a non-repeating shuffle of the dataset |

Startup 225–230 s (`runai_streamer` worked); fast-prefill compile markers present; KV cache 66.9 GiB.

**Dataset, measured in the pod** ([dataset_stats.json](results_run1_g4-48/dataset_stats.json)): 100,000 rows, 0 duplicates, 0 consecutive rows sharing a chat prefix. Prompt tokens with chat template: **mean 352**, p50 350, p90 468, p99 685, max 4,042; 5.4% over 512. Shared instruction prefix 98 tokens → measured prefix-cache hit rate **27.2–27.8%** on every step (predicted 27%). With the plain `s1:` format of their `content_str` column the mean is 326 tokens.

**Sanity checks:** top-1 token is Safe or Un on 100% of requests; both ids in the top-20 on 99.9%; p_unsafe AUC against their `is_violative` labels 0.61–0.63 (public model, not their fine-tune).

## 2. Closed loop (fixed concurrency, 60 s per point, run 1 on g4-48)

| c | req/s | p50 ms | p90 ms | p99 ms | GPU busy | GPU W | running |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 63.6 | 15.7 | 17.1 | 22.5 | 73% | 330 | 1 |
| 4 | 114.2 | 35.0 | 40.1 | 45.3 | 89% | 419 | 1.7 |
| 8 | 146.2 | 54.2 | 60.7 | 72.4 | 97% | 487 | 3.0 |
| 12 | 185.6 | 63.8 | 72.7 | 85.6 | 91% | 520 | 4.3 |
| 16 | 201.3 | 79.2 | 88.7 | 96.8 | 93% | 544 | 6.0 |
| 20 | 211.3 | 93.7 | 103.7 | 117.2 | 93% | 558 | 9.1 |
| 24 | 216.6 | 110.9 | 122.1 | 137.4 | 94% | 570 | 9.9 |
| 32 | 225.1 | 141.2 | 155.3 | 181.1 | 94% | 582 | 11.8 |
| 48 | 234.0 | 203.5 | 223.6 | 244.6 | 95% | 590 | 20.2 |
| 64 | 241.6 | 262.0 | 288.0 | 309.2 | 99% | 599 | 18.1 |
| 96 | 243.2 | 393.3 | 421.7 | 454.0 | 99% | 599 | 40.2 |
| 128 | 247.5 | 516.1 | 540.9 | 568.0 | 99% | 600 | 44.0 |
| 256 | 249.3 | 1024.0 | 1029.8 | 1282.1 | 100% | 600 | 63.4 (+66 waiting) |

Interpolated: **p50 = 70 ms → 192 req/s (c ≈ 14); p99 = 110 ms → 208 req/s (c ≈ 19); plateau 249 req/s.**

Cross-check of the second host (run 2, one GPU of the g4-384 node): c=8 162 req/s / p50 49 ms, c=16 191 / 84, c=32 224 / 143. Same ceiling as run 1; the −5…+11 % at low c is window-to-window prompt-mix noise. Read on run 2's points alone the SLO readouts would be 187 (p50 = 70) and 190 (p99 = 110) req/s, so quote them as 190 ± 5 and 200 ± 10.

## 3. Open loop (Poisson arrivals, 2 min per rate, run 2)

| Offered req/s | Served | p50 ms | p90 ms | p99 ms | running / waiting |
| --: | --: | --: | --: | --: | :-- |
| 120 | 119.8 | 41 | 61 | 80 | 2.0 / 0 |
| 160 | 159.9 | 60 | 89 | 119 | 4.3 / 0 |
| **175 (10-min soak)** | **175.2** | **69.7** | **104** | **144** | 5.3 / 0; 105,103 requests, 0 errors, flat |
| 190 | 190.4 | 83 | 130 | 185 | 5.3 / 0 |
| 215 | 215.3 | 119 | 190 | 277 | 8.7 / 0 |
| 240 | 240.6 | 271 | 430 | 571 | 33 / 0.7 (edge of capacity) |
| 280, 320 | overload | 7–15 s | | | queue grows to 1,700–2,700 |

Random arrivals cost ~10% versus fixed concurrency at the same p50, because of queueing variance: **p50 = 70 ms at ~175 req/s, p99 = 110 ms at ~150 req/s.**

## 4. Variants at 175 req/s (2 min each, run 2)

| Variant | p50 | p90 | p99 | Read |
| :--- | --: | --: | --: | :--- |
| Base (shuffled, `[s1]:` format, middleware on) | 69.7 | 104 | 144 | soak |
| File order | 69.4 | 106 | 148 | No cross-row prefix sharing in their file; order is irrelevant |
| **Plain `s1:` format** (their `content_str`, 326 tokens) | **57.9** | **85** | **119** | ~15% lower latency; worth ~15–20 req/s at p50 = 70 ms |
| Middleware bypassed | 72.0 | 110 | 149 | **Middleware cost: nil** (also c=16: 199 vs 198 req/s) |
| No `allowed_token_ids` | 69.7 | 105 | 140 | Nil |
| `/v1/completions` (hand-built chat wrapper) | 79.5 | 123 | 178 | Not comparable: the wrapper added 17 tokens (369 vs 352) |

## 5. Tie-back to our earlier synthetic spec (same server, `vllm bench serve`, 50 + 450 random tokens, 1 output token)

| c | This server (customer flags) | Our earlier "B" run (1 GPU, fast-prefill, nightly build) |
| --: | :--- | :--- |
| 8 | 105.9 req/s, p50 75 ms | 108.2 req/s, p50 74 ms |
| 16 | 121.8 req/s, p50 131 ms | 120.4 req/s |
| 64 | 130.5 req/s, p50 491 ms | 136.8 req/s |
| 128 | 139.2 req/s, p50 1,000 ms | ~136 req/s plateau |

The customer's build and flags (v0.30.0, `TRITON_ATTN`, 8k context) perform the same as our earlier configuration, so every number in the earlier comparison carries over unchanged. The whole gap between "136 req/s" and "190–240 req/s" is the prompt length: 500 synthetic tokens versus 352 real ones (256 computed after the cached prefix). Both runs sit on the same ~64k computed-tokens/s ceiling.

## 6. Reading

* **H1 confirmed.** On their real ~350-token prompts one GPU tops out at ~250 req/s and gives ~175–210 req/s at their latency targets, depending on how the load is offered. Their 190 is exactly what the GPU does; nothing in their stack (middleware, `allowed_token_ids`, client) costs measurable throughput.
* **Their upper figure, 240 req/s, is not reachable at p50 = 70 ms on this data** (240 req/s means c ≈ 64 and p50 ≈ 260 ms closed loop, or the edge of overload open loop). The band most likely spans prompt formats (the plain `s1:` format is 15% cheaper), checkpoints, or looser latency settings.
* **Where the ceiling comes from:** at c ≥ 64 the GPU sits at its 600 W cap and SM clocks fall from 2.39 to 2.27 GHz. ~256 computed tokens × 250 req/s ≈ 64k tokens/s, the same plateau as the synthetic run.
* **Not CPU:** load generator 0.02–0.06 cores, API server (`serve_main`) 0.3–0.5 cores, engine core 0.5–1.0 core (its single scheduler thread is pegged at c ≥ 128, but the GPU is at 99–100 % and the 600 W cap at the same points); no queue until the GPU saturates.
* **Prefix cache:** 27% hit rate as predicted (three 32-token blocks of the instruction prefix); no cross-conversation sharing exists in the file.

## 7. Consequences for the rest of the engagement

1. The G4-vs-v6e comparison must be redone on this prompt mix (352 tokens, 256 computed), not on the 500-token spec; the same harness now runs on either platform.
2. For an 8-GPU box the honest extrapolation is 8 × ~190–210 = 1,500–1,700 req/s at 70/110 ms — matching the customer's own 1,500–1,920 extrapolation at the low end — but the 8-GPU topology test (DP=8 on one node) is still owed, since they extrapolated and we measured one GPU too.
3. If they adopt the plain `s1:` line format (as in their `content_str`), they gain ~15% at no cost.

**Files:** [results_run1_g4-48/](results_run1_g4-48) (closed-loop sweep, Poisson 120), [results_run2_g4-384/](results_run2_g4-384) (cross-check, Poisson 160–320, soak, variants, tie-back). Each has `loadgen/steps.jsonl` (one JSON per step), per-request samples (latency, token count, score; no text), `gpu.csv`, `cpu.csv`, server logs. Verified: no prompt text in any result file; the `rbx-bench` namespace (and the data in its tmpfs) is deleted.
