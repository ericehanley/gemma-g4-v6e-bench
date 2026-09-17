# Additional vLLM Inference Optimizations for Gemma 4 31B on Cloud TPU v6e

## Overview

This document outlines advanced optimization strategies for serving **google/gemma-4-31B** on **Cloud TPU v6e (Trillium)** using the **vLLM** engine, tailored to Roblox's binary classification / moderation workload.

### Workload Profile & Baseline Context
* **Model**: `google/gemma-4-31B` (~31 billion parameters; 60 layers, 32 attention heads, 16 KV heads, hidden dimension 5376, head dimension 256).
* **Input**: 500 tokens (50 fixed TOS/safety policy prefix tokens + 450 prompt tokens).
* **Output**: 1 token (binary classification decision).
* **SLO Targets**: p90 End-to-End (E2E) latency $< 1{,}000\text{ ms}$ (strict constraint), identifying the $500\text{ ms}$ p90 knee.
* **Goal**: Maximize throughput (QPS) and price/performance ($QPS / \text{cost}$).
* **Hardware Constraint**: Cloud TPU v6e chips feature **32 GB HBM** per chip. In BF16, model weights occupy $\sim 62\text{ GB}$, making TP=1 unviable without quantization due to immediate OOM.

---

## 1. Native FP8 / W8A8 Quantization on Cloud TPU v6e (Trillium)

### Technical Mechanism
Cloud TPU v6e introduces dedicated hardware execution units for **FP8** arithmetic (`float8_e4m3fn` and `float8_e5m2`).

For Gemma 4 31B:
* **BF16 Baseline**: Model weights occupy $\sim 62\text{ GB}$ across the slice.
  * In **TP=2**, weights consume $\sim 29\text{--}30\text{ GB}$ per chip, leaving very narrow headroom ($\sim 1.5\text{--}2\text{ GB}$) for KV cache and activation buffers.
  * In **TP=4**, weights consume $\sim 15\text{ GB}$ per chip.
* **FP8 Quantized**: Model weights occupy $\sim 31\text{ GB}$ total across the slice.
  * In **TP=2**, weights drop to $\sim 15\text{ GB}$ per chip, liberating $>15\text{ GB}$ of HBM for high-concurrency KV caching and deep prefill batching.
  * **Unlocks Pure TP1-DP8**: At $\sim 31\text{ GB}$ in FP8, Gemma 4 31B can fit onto a single 32 GB chip (or via weight offloading / sub-block FP8), enabling **8 independent DP replicas** with **zero Inter-Chip Interconnect (ICI) communication overhead**.

### Manifest Configuration
```yaml
args:
- "vllm"
- "serve"
- "google/gemma-4-31B"
- "--quantization"
- "fp8"
- "--dtype"
- "bfloat16"
- "--tensor-parallel-size"
- "2"
- "--data-parallel-size"
- "4"
- "--gpu-memory-utilization"
- "0.95"
- "--max-model-len"
- "512"
- "--max-num-batched-tokens"
- "16384"
env:
- name: ENABLE_QUANTIZED_MATMUL_KERNEL
  value: "true"
- name: REQUANTIZE_WEIGHT_DTYPE
  value: "float8_e4m3fn"
- name: REQUANTIZE_BLOCK_SIZE
  value: "128"
```

### Expected Impact
* **Throughput**: $1.6\times\text{ to }2.0\times$ higher request throughput (QPS) on TPU v6e due to doubled MXU compute density.
* **Prefill Latency**: Prefill time for 500 tokens drops by $35\text{--}45\%$, pushing the $500\text{ ms}$ and $1{,}000\text{ ms}$ p90 latency boundaries much further out along the concurrency curve.

---

## 2. Direct Classification / Pooling Runner (Bypassing Autoregressive Decode)

### Technical Mechanism
Standard causal LLM generation executes:
1. **Prefill Phase**: Process 500 prompt tokens and compute KV cache.
2. **Decode Phase**: Run causal step, project to 262,144 vocabulary logits, sample next token, check EOS.
3. **KV Cache Lifecycle**: Allocate paged blocks, manage block tables, schedule prefill vs. decode priorities, free blocks upon termination.

For binary classification with 1-token output:
* Running an autoregressive decode loop adds scheduling latency and token sampling overhead across 262k logits.
* With vLLM's **Pooling / Classification Runner** (`--runner-type pooling` or the `/v1/classify` endpoint), the forward pass executes strictly once as an encoder-style prefill.
* Logits are projected directly to binary class probabilities (e.g. `allow` / `block`).
* **Zero KV cache blocks are allocated or tracked**, eliminating memory fragmentation and scheduling overhead.

### Manifest Configuration
```yaml
args:
- "vllm"
- "serve"
- "google/gemma-4-31B"
- "--runner-type"
- "pooling"
- "--tensor-parallel-size"
- "2"
- "--data-parallel-size"
- "4"
- "--dtype"
- "bfloat16"
- "--gpu-memory-utilization"
- "0.95"
- "--max-model-len"
- "512"
- "--max-num-batched-tokens"
- "16384"
```

### Expected Impact
* **Latency**: Reduces p90 E2E latency by **$30\%\text{ to }50\%$** by eliminating the second forward pass and token sampling pipeline.
* **Throughput**: Substantially higher request density. Because HBM is not reserved for KV cache blocks, `max-num-batched-tokens` can be safely increased to $16{,}384$ or $24{,}576$.

---

## 3. TPU Attention Precompilation & Bucket Padding Tuning

### Technical Mechanism
TPU execution with JAX/XLA compiles computation graphs for **static tensor dimensions**. In dynamic serving:
1. Batch sizes change dynamically based on incoming traffic.
2. Without explicit bucketing, vLLM either pads small batches up to large default bucket boundaries (wasting MXU FLOPs on padding tokens) or triggers expensive runtime JAX recompilations.

By configuring explicit compilation buckets and tuning token padding gaps:
* Attention kernels are precompiled ahead of time for the exact batch sizes expected under Roblox's traffic distribution.
* Padding waste is minimized by narrowing the bucket boundaries around the 500-token sequence size.

### Manifest Configuration
Set the following TPU environment variables documented in the [vLLM TPU reference](https://vllm--2806.org.readthedocs.build/projects/tpu/en/2806/environment_variables/):

```yaml
env:
- name: ATTN_CUSTOM_NUM_REQS_BUCKETS
  value: "1,2,4,8,16,32,64,128"
- name: VLLM_TPU_BUCKET_PADDING_GAP
  value: "512"
- name: MIN_TOKEN_BUCKET
  value: "16"
- name: VLLM_XLA_CHECK_RECOMPILATION
  value: "true"
```

### Expected Impact
* **Tail Latency Stability**: Eliminates latency jitter and tail spikes caused by dynamic shape re-tracing.
* **Compute Efficiency**: Reduces wasted MXU cycles on padding tokens by up to **$20\%$**, increasing throughput under sustained load.

---

## 4. Chunked Prefill Tuning (`--enable-chunked-prefill`)

### Technical Mechanism
Under high concurrency, large prefill batches (e.g. 16 requests $\times$ 500 tokens = 8,000 tokens) can monopolize TPU matrix multiply units for several hundred milliseconds, causing concurrent decode steps or new incoming requests to stall in the queue.

With `--enable-chunked-prefill`:
* Prefills are chunked into smaller slices (e.g. 512 tokens), co-scheduling prefill chunks alongside decode steps.
* This interleaving preserves decode latency while maintaining high MXU saturation.

### Manifest Configuration
```yaml
args:
- "vllm"
- "serve"
- "google/gemma-4-31B"
- "--enable-chunked-prefill"
- "--max-num-batched-tokens"
- "8192"
# ... other server flags ...
```

---

## Summary Comparison of Optimizations for Gemma 4 31B

| Optimization | Primary Benefit | Latency Impact | Throughput Impact | Implementation Effort |
| :--- | :--- | :--- | :--- | :--- |
| **FP8 Quantization (W8A8)** | Cuts weight memory in half; unlocks TP1-DP8 | $35\%\text{--}45\%$ reduction | $+60\%\text{--}90\%$ | Medium (Model calibration/quantization) |
| **Classification/Pooling Runner** | Bypasses decode & KV cache allocation | $30\%\text{--}50\%$ reduction | $+40\%\text{--}60\%$ | Medium (Model head adaptation) |
| **Attention Bucket Tuning** | Eliminates XLA shape recompilations & padding waste | Stabilizes p90/p99 tail | $+15\%\text{--}20\%$ | Low (Environment variables) |
| **Chunked Prefill** | Prevents prefill monopolization during traffic spikes | Smooths latency spikes | $+10\%\text{--}15\%$ | Low (CLI flag) |
