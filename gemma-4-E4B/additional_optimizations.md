# Additional vLLM Inference Optimizations for Gemma 4 E4B on Cloud TPU v6e

## Overview

This document details four advanced optimization strategies for serving the **Gemma 4 E4B** binary classification workload on **Cloud TPU v6e (Trillium)** using the **vLLM** engine. 

### Workload Profile & Baseline Context
* **Input**: 500 tokens (50 fixed TOS/safety policy prefix tokens + 450 prompt tokens).
* **Output**: 1 token (binary classification decision: `0`/`1`, `allow`/`block`, `yes`/`no`).
* **SLO Targets**: p90 End-to-End (E2E) latency $< 1{,}000\text{ ms}$ (strict constraint) with target identification of the $500\text{ ms}$ p90 threshold.
* **Goal**: Maximize throughput (QPS) and price/performance ($QPS / \text{cost}$).
* **Key Characteristic**: **99.8% prefill-dominated**. Autoregressive generation lasts exactly one step. Request holding times in the engine are virtually zero once prefill completes.

---

## 1. Direct Classification / Pooling Runner (Bypassing Autoregressive Decode)

### Technical Mechanism
Standard LLM serving frameworks (including vLLM in generative mode) route text generation requests through an autoregressive decode loop:
1. **Prefill Pass**: Process prompt tokens, build KV cache.
2. **Decode Pass(es)**: Run single-token decode step, sample token from vocabulary distribution ($>256{,}000$ logits in Gemma), check stop criteria (`max_tokens`, EOS).
3. **KV Cache Lifecycle**: Allocate paged blocks, manage block tables, schedule prefill vs. decode priorities, free blocks upon request termination.

For binary classification with a fixed 1-token output, running a full causal decode loop introduces unnecessary scheduling overhead, token sampling costs, and KV cache allocation churn.

Instead, vLLM supports a **Pooling / Classification Runner** (`--runner-type pooling` or the `/v1/score` / `/v1/classify` endpoints). When Gemma 4 E4B is fine-tuned with a classification head (or extracting pooled sequence logits):
* The forward pass executes strictly **once** as an encoder-style prefill step.
* Logits are projected directly to class probabilities (e.g., 2 classes).
* **No KV cache blocks are allocated or tracked**, and no autoregressive scheduling loop runs.

### Manifest Configuration
To configure vLLM in classification/pooling mode:
```yaml
containers:
- name: vllm-server
  image: vllm/vllm-tpu:nightly
  args:
  - "vllm"
  - "serve"
  - "google/gemma-4-E4B"
  - "--runner-type"
  - "pooling"
  - "--tensor-parallel-size"
  - "1"
  - "--data-parallel-size"
  - "8"
  - "--dtype"
  - "bfloat16"
  - "--gpu-memory-utilization"
  - "0.95"
  - "--max-model-len"
  - "512"
  - "--max-num-batched-tokens"
  - "16384"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "8000"
```

### Expected Impact
* **Latency**: Reduces p90 E2E latency by **30% to 50%** by eliminating the second forward pass and token sampling pipeline.
* **Throughput**: Significantly higher request density. Because memory is not reserved for KV cache blocks, `max-num-batched-tokens` can be safely increased (e.g., to $16{,}384$ or $24{,}576$), allowing up to 32–48 requests to be evaluated simultaneously in a single forward pass.
* **Trade-offs**: Requires the model checkpoint to expose classification/pooling outputs (e.g., `AutoModelForSequenceClassification`) or client querying via classification endpoints rather than `/v1/completions`.

---

## 2. TPU JAX/XLA Attention Compilation & Batch Bucketing Tuning

### Technical Mechanism
TPU execution with JAX/XLA compiles computation graphs for **static tensor dimensions**. In dynamic serving:
1. Batch sizes change dynamically based on incoming traffic.
2. Without explicit bucketing, vLLM either pads small batches up to large default bucket boundaries (wasting TPU Matrix Multiply Unit [MXU] FLOPs on padding tokens) or triggers expensive runtime JAX recompilations.

By configuring explicit compilation buckets and tuning token padding gaps:
* Attention kernels are precompiled ahead of time for the exact batch sizes expected under Roblox's traffic distribution.
* Padding waste is minimized by narrowing the bucket boundaries around the 500-token sequence size.

### Environment & Manifest Configuration
Set the following TPU environment variables documented in the [vLLM TPU reference](https://vllm--2806.org.readthedocs.build/projects/tpu/en/2806/environment_variables/):

```yaml
env:
# Precompile attention for specific request batch sizes matching production concurrency
- name: ATTN_CUSTOM_NUM_REQS_BUCKETS
  value: "1,2,4,8,16,32,64,128"
# Tune token-bucket padding gap (e.g. 512 tokens) to align with ~500-token prompts
- name: VLLM_TPU_BUCKET_PADDING_GAP
  value: "512"
# Set minimum token bucket to 16
- name: MIN_TOKEN_BUCKET
  value: "16"
# Ensure XLA recompilation triggers an immediate alert during benchmarking
- name: VLLM_XLA_CHECK_RECOMPILATION
  value: "true"
```

### Expected Impact
* **Latency Consistency**: Eliminates latency spikes (tail jitter) caused by dynamic shape re-tracing. P99 and P90 latencies align much closer to P50.
* **Compute Efficiency**: Reduces wasted MXU cycles on pad tokens by up to **20%**, allowing the TPU to process more requests per second at identical power consumption.
* **Trade-offs**: Slightly increases initial server startup and warmup compilation time ($+1\text{ to }3\text{ minutes}$ during pod initialization).

---

## 3. Native FP8 / W8A8 Quantization on Cloud TPU v6e (Trillium)

### Technical Mechanism
Cloud TPU v6e (Trillium) introduces native hardware execution units for **FP8** arithmetic (`float8_e4m3fn` and `float8_e5m2` formats). 

For Gemma 4 E4B:
* **BF16 Baseline**: Model weights occupy $\sim 8\text{ GB}$ of HBM ($4\text{B parameters} \times 2\text{ bytes}$).
* **FP8 Quantized**: Model weights occupy $\sim 4\text{ GB}$ of HBM ($4\text{B parameters} \times 1\text{ byte}$).

Because TPU v6e MXUs achieve double the raw TFLOPs in FP8 relative to BF16, matrix multiplications across attention projections and feed-forward networks (MLP layers) execute in approximately half the hardware time.

### Manifest Configuration
Enable FP8 quantization either through model checkpointing or vLLM TPU quantized kernels:

```yaml
args:
- "vllm"
- "serve"
- "google/gemma-4-E4B"
- "--quantization"
- "fp8"
- "--dtype"
- "bfloat16"
# ... other server flags ...
env:
- name: ENABLE_QUANTIZED_MATMUL_KERNEL
  value: "true"
- name: REQUANTIZE_WEIGHT_DTYPE
  value: "float8_e4m3fn"
- name: REQUANTIZE_BLOCK_SIZE
  value: "128"
```

### Expected Impact
* **Throughput**: Up to **$1.6\times\text{ to }1.9\times$ higher request throughput (QPS)** on TPU v6e due to doubled compute density.
* **Prefill Latency**: 500-token prefill step latency drops by **35–45%**, moving the p90 latency knee further out along the concurrency curve (reaching the $500\text{ ms}$ threshold only at significantly higher concurrency).
* **Trade-offs**: Requires calibrating quantization scales or using a pre-quantized checkpoint to verify that classification accuracy/recall on toxicity edge cases is fully preserved.

---

## 4. Sub-Slice Sizing and Pure TP=1 Pod Scaling (Maximizing Price/Performance)

### Technical Mechanism
A key objective in the benchmarking instructions is:
> *"The overall goal is to maximize price/performance which generally means the less chips we have to use the better."*

For a small model like Gemma 4 E4B ($\sim 4\text{B}$ active parameters, $\sim 8\text{ GB}$ footprint):
1. **Zero Inter-Chip Communication Overhead in TP=1**:
   * In TP=2, TP=4, or TP=8, every attention layer and MLP layer performs `All-Reduce` collectives across the TPU Inter-Chip Interconnect (ICI).
   * For small matrix dimensions, the latency of ICI communication synchronization can approach or exceed the actual computation time saved by sharding across chips.
   * In **TP=1**, each chip operates completely independently as a standalone inference engine with **0% communication overhead**.
2. **Sub-Slice Architecture**:
   * Instead of procuring full 8-chip slices (`tpu-v6e-slice`, topology `2x4`), Roblox can deploy independent replicas on **single-chip slices (`tpu-v6e-slice`, topology `1x1`)** or **4-chip slices (`2x2`)**.
   * Single-chip instances provide linear scaling: $N$ chips provide $N \times \text{throughput}$ with no inter-pod dependency, simpler Kubernetes pod bin-packing, and no multi-host synchronization.

### Recommended Production Topology
```mermaid
graph TD
    subgraph "Sub-Slice Topology (Max Price/Performance)"
        LB[Roblox API Gateway / Load Balancer]
        R1[TPU v6e-1: Replica 1 (TP=1)]
        R2[TPU v6e-1: Replica 2 (TP=1)]
        R3[TPU v6e-1: Replica 3 (TP=1)]
        R4[TPU v6e-1: Replica 4 (TP=1)]
        LB -->|Round Robin / Least Conn| R1
        LB -->|Round Robin / Least Conn| R2
        LB -->|Round Robin / Least Conn| R3
        LB -->|Round Robin / Least Conn| R4
    end
```

### Deployment Configuration for Single-Chip Slices
```yaml
nodeSelector:
  cloud.google.com/gke-tpu-accelerator: "tpu-v6e-slice"
  cloud.google.com/gke-tpu-topology: "1x1"
resources:
  limits:
    google.com/tpu: "1"
  requests:
    google.com/tpu: "1"
```

### Expected Impact
* **Price/Performance**: Delivers the **lowest cost per million classifications**. Benchmarking across the four manifests is expected to show that `tp1-dp8` matches or outperforms `tp8-dp1` in total throughput while offering superior fault isolation (failure of 1 chip affects only 1 replica rather than the entire 8-chip job).
* **Autoscaling Agility**: Single-chip pods boot, initialize, and compile faster than 8-chip mesh topologies, allowing Horizontal Pod Autoscalers (HPA) to scale dynamically with chat traffic surges.

---

## 5. Note on KV Cache & Prefix Warmup in Benchmarking vs. Production

### Why Manual Curl Warmup is Not Needed in the Benchmarking Manifests
* **Server-Side JAX/XLA Compilation**: During initialization on TPU, vLLM automatically executes dummy graph execution passes (`profile_run`) before declaring readiness. Port `8000` and the `/health` endpoint do not return HTTP 200 until this compilation warmup finishes.
* **Client-Side Benchmark Warmup**: `vllm bench serve` natively includes `--num-warmups 20`. Before recording timing metrics, the benchmark client dispatches 20 warmup requests using the exact synthetic dataset prompt (including the 50-token prefix). This naturally primes the prefix cache across all data-parallel replicas.
* **Production Relevance**: In a live production deployment (serving user traffic rather than `vllm bench serve`), adding a Kubernetes `readinessProbe` or startup lifecycle hook that issues a warmup request with the fixed 50-token TOS policy ensures that 100% of initial production requests hit the prefix cache immediately without cold-start jitter.

---

## Optimization Comparison Summary

| Optimization | Target Metric | Estimated Latency Improvement | Estimated Throughput Gain | Implementation Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **#1 Direct Classification / Pooling** | Latency & QPS | $30\% - 50\%$ E2EL reduction | $+40\% - 60\%$ | Medium (Requires model export as sequence classifier) |
| **#2 TPU Attention & Bucket Tuning** | Latency Jitter (p90/p99) | Stabilizes tail latency | $+15\% - 20\%$ | Low (Environment variable tuning in existing manifests) |
| **#3 Native FP8 Quantization** | Max QPS & Compute Density | $35\% - 45\%$ prefill reduction | $+60\% - 90\%$ | Medium (Model quantization & accuracy validation) |
| **#4 Pure TP=1 Sub-Slice Sizing** | Price / Performance | Zero ICI communication wait | Linear horizontal scaling | Low (Kubernetes nodeSelector & resource resizing) |
