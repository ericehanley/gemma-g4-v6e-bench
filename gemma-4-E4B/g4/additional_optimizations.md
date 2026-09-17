# Additional vLLM Inference Optimizations for Gemma 4 E4B on G4 (RTX PRO 6000 Blackwell)

## Overview

This document outlines advanced optimization strategies for serving **google/gemma-4-E4B** on **GCP G4 instances (NVIDIA RTX PRO 6000 Blackwell Server Edition)** using the **vLLM** engine, for Roblox's binary classification / moderation workload.

Every number below is anchored to **measurements from the actual benchmark run** on `g4-standard-384` (8 × RTX PRO 6000, 96 GB GDDR7 each).

### Workload Profile & Baseline Context
* **Model**: `google/gemma-4-E4B` (8.0 B effective params BF16; 42 layers, of which `num_kv_shared_layers: 18` share KV — only **24 layers** actually allocate cache).
* **Input**: 500 tokens (50 fixed TOS/safety policy prefix + 450 prompt tokens).
* **Output**: 1 token.
* **SLO Targets**: p90 E2E $< 1{,}000\text{ ms}$ (strict), with the $500\text{ ms}$ knee also identified.
* **Baseline achieved**: TP1-DP8, BF16 — **643.0 req/s @ c=256** under the 500 ms SLO; **649.9 req/s @ c=512** under the 1 s SLO.

### Measured Hardware Envelope

| Quantity | Measured value |
| :--- | :--- |
| GDDR7 per GPU | 96 GB (0.90 utilization → 86.4 GiB usable) |
| Model weights @ BF16, TP=1 | ~16 GiB / GPU |
| Remaining KV cache | **68.1 GiB** |
| KV capacity | **1,220,612 tokens** |
| Concurrent 512-token sequences per replica | **~2,384** |
| `--max-num-seqs` actually configured | **512** |
| **KV headroom over the configured limit** | **≈ 4.7×** (and ≈ 2,384× over a single sequence) |

> [!IMPORTANT]
> **E4B on G4 is the exact inverse of the 31B case.** For 31B, KV cache is the binding constraint (61 sequences/replica). For E4B, KV cache is **massively over-provisioned** — 68 GiB sitting largely idle. The evidence: measured throughput sits on a **flat plateau of 648–651 req/s across c=256 → c=1024** (0.5% variance across a 4× concurrency range). Adding concurrency buys *nothing*. The system is saturated on **compute and per-step scheduling overhead**, not memory.
>
> Consequently, **any optimization that merely frees memory is worthless here.** Optimizations must reduce work-per-request or work-per-step.

> [!NOTE]
> RTX PRO 6000 Blackwell Server Edition has **no NVLink** (PCIe Gen5 only). TP=1 / DP=8 is retained throughout — a tensor-parallel all-reduce would cross PCIe on every layer.

---

## 1. FP8 / NVFP4 Quantization on Blackwell (sm120) — a *Compute* Play, Not a Memory Play

### Technical Mechanism
The RTX PRO 6000 is a **Blackwell sm120** part with 5th-gen Tensor Cores supporting native **FP8 (`e4m3`)** and hardware block-scaled **NVFP4**.

With a 500-token prompt and a 1-token output, virtually 100% of the work is **prefill** — a large, dense, compute-bound GEMM sweep. This is the ideal shape for low-precision Tensor Cores:
* FP8 roughly **doubles Tensor Core matmul throughput** versus BF16.
* It halves the weight bytes streamed from GDDR7 per forward pass (16 GiB → 8 GiB), which matters because prefill at this batch size is partly bandwidth-bound.

> [!WARNING]
> Do **not** justify quantization here on memory savings. Freeing 8 GiB of an already-idle 68 GiB pool changes nothing. The entire benefit is arithmetic throughput. This is the opposite of the 31B case, where the memory reclaim *was* the whole point.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-E4B"
- "--quantization"
- "fp8"                      # or "modelopt_fp4" for NVFP4 checkpoints
- "--tensor-parallel-size"
- "1"
- "--data-parallel-size"
- "8"
- "--gpu-memory-utilization"
- "0.90"
- "--max-model-len"
- "512"
```
Leave `--kv-cache-dtype` at BF16 — FP8 KV would only deepen an already-unused pool while adding a quantize/dequantize step on the critical path.

### Expected Impact
* **Throughput**: $+50\%$ to $+80\%$ (≈ 975 – 1,170 req/s), since the plateau is compute-bound and this directly lifts compute.
* **Latency**: $25\text{--}40\%$ p90 reduction, pushing the 500 ms knee well past c=256.
* **Risk**: Medium. Requires accuracy validation of the classification decision against the BF16 baseline.

---

## 2. Cap the KV Cache and Spend the Memory on CUDA Graphs + Prefill Width

### Technical Mechanism
68.1 GiB of KV cache for a workload that needs at most 512 × 512 = 262,144 tokens (≈ 14.6 GiB) per replica is a **4.7× over-allocation**, and vLLM sizes the pool greedily from `--gpu-memory-utilization`. That memory is not free of cost — a larger block pool means a larger block table, more allocator bookkeeping per step, and worse cache locality in the scheduler's hot loop.

Explicitly capping KV lets the remainder go to things that actually help a compute-bound, high-QPS server:
* **Full CUDA graph capture.** At 8 B params with 1-token outputs, per-step kernel-launch and Python dispatch overhead is a meaningful fraction of step time. Full-graph mode replays the whole step as one graph. This *requires* spare memory to capture into.
* **Wider prefill batches** via a larger `--max-num-batched-tokens`, improving Tensor Core occupancy per step.
* **Async scheduling**, which overlaps the scheduler's next-step planning with the current step's GPU execution — directly attacking the per-step overhead that defines this plateau.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-E4B"
- "--kv-cache-memory"
- "21474836480"              # 20 GiB: ~358k tokens, still ~1.4x over the 512-seq need
- "--max-num-batched-tokens"
- "16384"                    # up from 8192; 32 x 500-token prefills per step
- "--max-num-seqs"
- "512"
- "--async-scheduling"
- "--compilation-config"
- '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

### Expected Impact
* **Throughput**: $+10\%$ to $+20\%$. Modest but cheap, and it targets exactly the bottleneck the plateau identifies.
* **Latency**: Lower and more stable p90 — graph replay removes launch-time jitter.
* **Risk**: Low. Validate that full-graph capture succeeds in the logs; vLLM falls back to piecewise if it cannot capture.

---

## 3. Disable the Multimodal Towers

### Technical Mechanism
Gemma 4 E4B is **multimodal** (vision + audio encoders); this workload is pure text. vLLM still loads the encoder weights, runs a multimodal profiling/warmup pass — **measured at 196 seconds** in our run — and reserves worst-case multimodal activation scratch.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-E4B"
- "--limit-mm-per-prompt"
- '{"image": 0, "audio": 0}'
```

### Expected Impact
* **Startup**: **−196 s** cold start, measured. On a model whose weights load in seconds, the multimodal warmup is the *dominant* component of startup time.
* **Throughput**: ~Neutral. The freed memory is not the constraint here (see Section 2).
* **Why it still matters**: G4 capacity is predominantly **Spot**. A 196 s recovery penalty on every preemption is a real availability cost, and it throttles how quickly you can autoscale into a traffic spike.
* **Risk**: None for a text-only workload.

---

## 4. Disable Prefix Caching

### Technical Mechanism
The benchmark ran with vLLM's default `--enable-prefix-caching`. The prompts share only a **50-token** fixed prefix out of 500 — **10%**. The server logs report `kv lcm block sizes 32`, so that 50-token prefix spans exactly **one** fully-aligned shareable block (tokens 0–31); tokens 32–49 land in a block that also contains random content and can never be reused. The other 450 tokens are random per request and yield **zero** hits.

The server nonetheless hashes every block of every request, probes the cache, and maintains LRU eviction state. Prefix caching also **retains completed requests' blocks** instead of freeing them, so the pool is continuously churning through a working set it can never reuse.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-E4B"
- "--no-enable-prefix-caching"
```

### Expected Impact
* **Throughput**: $+2\%$ to $+8\%$, from removing per-block hashing and LRU maintenance from the request path. Note that unlike the 31B case, the *memory* reclaimed is worthless here — only the CPU-side bookkeeping saving counts, which is why the estimate is conservative.
* **Latency**: Mainly p99; removes allocator churn from the tail.
* **Risk**: Low. Sacrifices a 3-block prefix hit worth <1% of prefill.

> [!NOTE]
> A dedicated diagnostic job (`gemma-4-e4b-g4-diag`) was run specifically to test whether prefix-cache churn explains the observed gap between **burst** throughput (900 req/s over a 1.1 s run) and **sustained** throughput (650 req/s over a 30 s run). See the benchmark comparison writeup for the outcome. If churn is confirmed as the mechanism, the upside of this section is substantially larger than the $2\text{--}8\%$ quoted above.

---

## 5. Pooling / Classification Runner (Skip Generation Entirely)

### Technical Mechanism
The workload emits **exactly one token**. The generative path is heavy overhead for that: KV block allocation, a prefill step, a *separate* decode step, sampling (top-p/top-k via FlashInfer, which our logs show is autotuned over token buckets), detokenization, and streaming response assembly.

vLLM's pooling runner collapses this to a single forward pass returning a classification logit:
* **No KV cache at all** — removes the entire subsystem that Sections 2 and 4 are trying to tune around.
* **No second scheduler step** per request. Given that E4B's plateau is explicitly a per-step-overhead ceiling, halving the number of steps per request is the most direct possible attack on it.
* No sampler, no detokenizer.

### Manifest Configuration
```yaml
args:
- "serve"
- "google/gemma-4-E4B"
- "--runner"
- "pooling"
- "--convert"
- "classify"
- "--max-num-batched-tokens"
- "32768"
```
Clients call `/classify` instead of `/v1/completions`.

### Expected Impact
* **Throughput**: $+50\%$ to $+80\%$ — likely the single largest win for E4B, larger in relative terms than for 31B, precisely because E4B is step-overhead-bound rather than memory-bound.
* **Latency**: $30\text{--}50\%$ p90 reduction.
* **Risk**: **High — architectural, not a flag.** Requires a calibrated classification head and a client contract change, and forfeits free-text rationales.

---

## Summary Comparison of Optimizations for Gemma 4 E4B on G4

| Optimization | Attacks | Throughput Impact | Latency Impact | Effort / Risk |
| :--- | :--- | :--- | :--- | :--- |
| **1. FP8 / NVFP4 quantization** | Compute (prefill GEMMs) | $+50\%\text{--}80\%$ | $-25\%\text{--}40\%$ | Medium (accuracy validation) |
| **2. Cap KV + CUDA graphs + async sched** | Per-step overhead | $+10\%\text{--}20\%$ | Lower jitter | **Low (flags only)** |
| **3. Disable multimodal towers** | Startup / Spot recovery | ~Neutral | Neutral | **Trivial (one flag, zero risk)** |
| **4. Disable prefix caching** | CPU-side bookkeeping | $+2\%\text{--}8\%$ | Tightens p99 | **Low (one flag)** |
| **5. Pooling / classify runner** | Steps-per-request | $+50\%\text{--}80\%$ | $-30\%\text{--}50\%$ | High (model head + API change) |

### Recommended Rollout Order
1. **Sections 3 + 4** — two flags, no accuracy risk. Ship immediately; Section 3 alone pays for itself the first time a Spot node is preempted.
2. **Section 2** — flags only, directly targets the measured plateau. Verify full CUDA-graph capture in the logs.
3. **Section 1 (FP8)** — the main throughput unlock. Budget accuracy-validation time.
4. **Section 5 (pooling)** — highest ceiling, but only worth it if the moderation decision is genuinely binary and you control the clients.

> [!WARNING]
> Sections 1, 2 and 5 all attack overlapping bottlenecks and are **not additive**. Once compute is doubled by FP8, the per-step overhead addressed in Section 2 becomes a *larger* share of the remaining time, not a smaller one — so re-measure after each step rather than projecting the product of the percentages. Section 5 subsumes most of Section 2.

### Contrast with the 31B Recommendations
| | **31B on G4** | **E4B on G4** |
| :--- | :--- | :--- |
| Binding constraint | **KV cache** (61 seqs/replica) | **Compute + per-step overhead** |
| Evidence | Throughput knee tracks KV exhaustion | Flat 648–651 req/s plateau over a 4× concurrency range |
| Why quantization helps | Frees 29–44 GiB → 2–4× sequence capacity | Doubles Tensor Core throughput; freed memory is *useless* |
| Value of freeing memory | **Very high** | **Zero** |
| Best single flag-only change | `--no-enable-prefix-caching` | `--async-scheduling` + full CUDA graphs |
