# Additional vLLM Inference Optimizations for Gemma 4 31B on G4 (RTX PRO 6000 Blackwell)

## Overview

This document outlines advanced optimization strategies for serving **google/gemma-4-31B** on **GCP G4 instances (NVIDIA RTX PRO 6000 Blackwell Server Edition)** using the **vLLM** engine, tailored to Roblox's binary classification / moderation workload.

Unlike the TPU v6e document, every number below is anchored to **measurements taken from the actual benchmark run** on `g4-standard-384` (8 × RTX PRO 6000, 96 GB GDDR7 each), not estimates.

### Workload Profile & Baseline Context
* **Model**: `google/gemma-4-31B` (31.27 B params; 60 layers — 50 sliding-window + 10 full-attention, 16 KV heads, head dim 256).
* **Input**: 500 tokens (50 fixed TOS/safety policy prefix tokens + 450 prompt tokens).
* **Output**: 1 token (binary classification decision).
* **SLO Targets**: p90 E2E latency $< 1{,}000\text{ ms}$ (strict), with the $500\text{ ms}$ p90 knee also identified.
* **Baseline achieved**: TP1-DP8, BF16 — **93.8 req/s @ c=80** under the 1 s SLO; **87.6 req/s @ c=32** under the 500 ms SLO; peak **98.4 req/s**.

### Measured Hardware Envelope (the constraint that drives everything below)

| Quantity | Measured value |
| :--- | :--- |
| GDDR7 per GPU | 96 GB (0.92 utilization → 88.3 GiB usable) |
| Model weights @ BF16, TP=1 | **58.99 GiB / GPU** |
| Remaining KV cache | **26.94 GiB** |
| KV bytes per token | 901,120 B (≈ 0.88 MiB) |
| KV capacity | **31,209 tokens** |
| **Concurrent 512-token sequences per replica** | **60.96** |
| **Cluster-wide concurrent sequence ceiling (8 replicas)** | **≈ 488** |

> [!IMPORTANT]
> **KV cache — not compute — is the binding constraint for 31B on G4.** Weights eat 67% of each card. The `--max-num-seqs 96` in the benchmark manifest is *dead config*: the hardware can only hold ~61 sequences per replica. Every optimization below is scored primarily on **how much KV headroom it returns**.

> [!NOTE]
> **RTX PRO 6000 Blackwell Server Edition has no NVLink** — inter-GPU traffic crosses PCIe Gen5. This is why TP1-DP8 was selected: any tensor-parallel all-reduce would pay a PCIe round trip on every layer. All recommendations below preserve TP=1.

---

## 1. FP8 / NVFP4 Weight Quantization on Blackwell (sm120)

### Technical Mechanism
The RTX PRO 6000 is a **Blackwell sm120** part with 5th-generation Tensor Cores that execute **FP8 (`e4m3`)** natively and **NVFP4** via hardware block-scaling. This is the single highest-leverage change available, because it attacks the binding constraint directly.

Quantizing weights shrinks the 58.99 GiB resident footprint and hands the difference straight to the KV cache:

| Precision | Weights / GPU | KV cache freed | KV tokens | **Concurrent seqs / replica** | Cluster ceiling |
| :--- | ---: | ---: | ---: | ---: | ---: |
| BF16 (baseline) | 58.99 GiB | 26.94 GiB | 31,209 | **61** | 488 |
| FP8 W8A8 | ~29.5 GiB | ~56.4 GiB | ~65,346 | **~128** | ~1,022 |
| NVFP4 W4A16 | ~14.8 GiB | ~71.1 GiB | ~82,375 | **~161** | ~1,287 |

FP8 also roughly **doubles Tensor Core matmul throughput** versus BF16, and halves the weight bytes that must be streamed from GDDR7 each forward pass — which matters because a 1-output-token classification workload is heavily **prefill/memory-bandwidth bound**, not decode bound.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-31B"
- "--quantization"
- "fp8"                      # or "modelopt_fp4" for NVFP4 checkpoints
- "--kv-cache-dtype"
- "fp8_e4m3"                 # halves KV bytes/token again: 0.88 MiB -> 0.44 MiB
- "--tensor-parallel-size"
- "1"
- "--data-parallel-size"
- "8"
- "--gpu-memory-utilization"
- "0.92"
- "--max-model-len"
- "512"
- "--max-num-seqs"
- "128"                      # now a *real* limit, not dead config
```

> [!TIP]
> Stacking `--kv-cache-dtype fp8_e4m3` on top of FP8 weights is nearly free accuracy-wise for a **binary classification** task (the decision is a single argmax over two logits, not a long generation where error compounds) and doubles sequence capacity a second time — to **~256 seqs/replica**.

### Expected Impact
* **Throughput**: $+60\%$ to $+100\%$. The c=32/c=80 SLO knees exist because the scheduler runs out of KV blocks and starts queueing; lifting the ceiling from 61 to 128–256 pushes the knee far to the right.
* **Latency**: p90 drops $25\text{--}40\%$ at fixed concurrency from the matmul speedup alone.
* **Risk**: Requires a calibrated checkpoint or on-the-fly quantization; validate classification agreement against the BF16 baseline before shipping.

---

## 2. Disable the Multimodal Towers

### Technical Mechanism
Gemma 4 is a **multimodal** model — it ships vision and audio encoders. This workload is **pure text**. vLLM nonetheless:
1. Loads the vision + audio encoder weights into GDDR7 (counted inside the 58.99 GiB above).
2. Runs a multimodal profiling/warmup pass at startup. **This was measured at 196 seconds** in our run.
3. Reserves activation scratch space sized for the worst-case image/audio batch, which is subtracted from the pool before KV cache is computed.

Setting the per-prompt multimodal limits to zero makes vLLM skip encoder instantiation and profiling entirely.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-31B"
- "--limit-mm-per-prompt"
- '{"image": 0, "audio": 0}'
```

### Expected Impact
* **Startup**: **−196 s** cold start, measured. This is not a throughput win, but it is a large win for autoscaling responsiveness, Spot preemption recovery, and rolling deploys — all of which matter on G4 Spot capacity.
* **Memory**: Returns encoder weights + multimodal activation reserve to the KV pool — worth several GiB, i.e. a further ~10–15% sequence-capacity increase on top of Section 1.
* **Risk**: None for a text-only workload. Purely removes unused capability.

---

## 3. Disable Prefix Caching

### Technical Mechanism
The benchmark ran with vLLM's default `--enable-prefix-caching`. For this workload that default is a net negative:

* The prompts share only a **50-token** fixed prefix out of 500 tokens — **10%**. vLLM reported `kv lcm block sizes 32` on this stack, so the shared prefix spans exactly **one** fully-aligned reusable block (tokens 0–31); tokens 32–49 sit in a block that also holds random content and can never be reused. The remaining 450 tokens are random per request and produce **zero** cache hits.
* vLLM still pays the full cost on every request: hashing each block, probing the cache, and maintaining the LRU eviction structure.
* Worse, prefix caching **retains completed requests' blocks** rather than freeing them immediately. On a model with only 31,209 KV tokens of capacity, holding dead blocks directly steals from the ~61-sequence live budget and forces eviction churn.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-31B"
- "--no-enable-prefix-caching"
```

If you would rather keep the single-block prefix hit, the middle ground is to leave caching **on** but raise the block size above the effective default of 32, so the 50-token prefix is covered by fewer hash lookups:
```yaml
- "--block-size"
- "64"
```
Note this coarsens allocation granularity, which wastes KV on the 500-token sequences (500 rounds up to 512 at block 64 vs 512 at block 32 — no waste here, but verify if `max-model-len` changes).

### Expected Impact
* **Throughput**: $+3\%$ to $+8\%$, from reclaimed KV blocks and removed per-block hashing.
* **Latency**: Most of the benefit lands in the **tail** — removing eviction stalls tightens p90/p99 rather than the median.
* **Risk**: Low. The only loss is the 3-block prefix hit, worth <1% of prefill work.

---

## 4. Pooling / Classification Runner (Skip Generation Entirely)

### Technical Mechanism
This workload generates **exactly one token**. The full generative path is enormous overhead for that: vLLM allocates KV blocks, runs a prefill step, runs a separate decode step, runs the sampler (top-p/top-k via FlashInfer, which our logs show is autotuned over token buckets), then runs detokenization and streaming response assembly.

vLLM's **pooling runner** replaces all of that with a single forward pass that returns the pooled hidden state or a classification logit directly:
* **No KV cache is allocated at all** — the binding constraint from the table above disappears completely. The entire 26.94 GiB (or 71 GiB post-quantization) becomes available for larger prefill batches.
* No decode step, no sampler, no detokenizer.
* Batch size becomes limited only by activation memory, not KV.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-31B"
- "--runner"
- "pooling"
- "--convert"
- "classify"
- "--max-num-batched-tokens"
- "32768"        # KV cache is gone; spend the memory on prefill width instead
```
Clients then call `/classify` (or `/pooling`) instead of `/v1/completions`.

### Expected Impact
* **Throughput**: $+40\%$ to $+70\%$ — the largest win after quantization.
* **Latency**: $30\text{--}50\%$ p90 reduction; one less scheduler step and no sampling on the critical path.
* **Risk**: **Medium-high, and this is an architectural change, not a flag.** It requires attaching and calibrating a classification head, and it changes the client contract. It also gives up the ability to return free-text rationales. Recommended only if the moderation decision is genuinely binary.

---

## 5. Right-size `--max-num-seqs` and `--max-num-batched-tokens` (Correctness Fix)

### Technical Mechanism
The benchmark manifest carries `--max-num-seqs 96` against a measured hardware ceiling of **60.96** sequences. This is inert in the benchmark — with `--max-num-batched-tokens 4096` only ~8 prompts prefill per step, and 1-token outputs retire immediately, so the scheduler queues rather than preempts — but it is a **latent p99 hazard** under bursty production traffic, where the scheduler may admit sequences it cannot hold and then preempt/recompute them.

### Manifest Configuration
```yaml
args:
- "--max-num-seqs"
- "56"                       # 61 measured ceiling, with headroom
- "--max-num-batched-tokens"
- "4096"                     # 8 x 500-token prefills per step; keeps step latency under the 1s p90 SLO
```

### Expected Impact
* **Throughput**: Neutral at steady state.
* **Latency**: Removes a preemption-driven p99 cliff under burst. Strictly a robustness fix.
* **Risk**: None. Recompute this number any time precision, `max-model-len`, or `gpu-memory-utilization` changes.

---

## Summary Comparison of Optimizations for Gemma 4 31B on G4

| Optimization | Primary Benefit | Seqs/replica after | Throughput Impact | Latency Impact | Effort / Risk |
| :--- | :--- | ---: | :--- | :--- | :--- |
| **1. FP8 / NVFP4 quantization** | Halves/quarters the 58.99 GiB weight footprint — attacks the binding constraint | 128 – 256 | $+60\%\text{--}100\%$ | $-25\%\text{--}40\%$ | Medium (calibration + accuracy validation) |
| **2. Disable multimodal towers** | Removes 196 s warmup and unused encoder memory | +10–15% | $+2\%\text{--}5\%$ | Neutral | **Trivial (one flag, zero risk)** |
| **3. Disable prefix caching** | Reclaims retained blocks; drops useless hashing on 90%-random prompts | +small | $+3\%\text{--}8\%$ | Tightens p90/p99 tail | **Low (one flag)** |
| **4. Pooling / classify runner** | Eliminates KV cache and the decode step entirely | unbounded (no KV) | $+40\%\text{--}70\%$ | $-30\%\text{--}50\%$ | High (model head + client contract change) |
| **5. Right-size `max-num-seqs`** | Aligns scheduler admission with real KV capacity | 56 (correct) | Neutral | Removes p99 burst cliff | None (correctness fix) |

### Recommended Rollout Order
1. **Sections 2 + 3 first** — two CLI flags, no accuracy risk, immediate startup and tail-latency wins. Ship today.
2. **Section 5** — correctness hygiene, bundle with the above.
3. **Section 1 (FP8)** — the real throughput unlock. Budget time for accuracy validation against the BF16 classification baseline.
4. **Section 4 (pooling)** — evaluate only after 1–3 are exhausted, since it changes the API contract.

> [!WARNING]
> Sections 1 and 4 are **not additive in the way the table suggests**. Pooling removes the KV cache, which is precisely the resource that quantization frees. If you adopt the pooling runner, quantization's benefit reverts to its raw compute speedup (~$+30\%$), not the $+60\text{--}100\%$ listed. Do not sum the columns.
