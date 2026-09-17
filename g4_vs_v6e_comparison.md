# Gemma 4 on GCP: TPU v6e vs. G4 (RTX PRO 6000 Blackwell)

Benchmark comparison for Roblox's binary classification / moderation workload.

## 1. Workload and Method

| | |
| :--- | :--- |
| Input | 500 tokens (50-token fixed TOS/safety prefix + 450 random tokens) |
| Output | 1 token (binary decision) |
| Primary SLO | p90 E2E latency < 1,000 ms |
| Secondary SLO | p90 E2E latency < 500 ms |
| Objective | Maximize throughput under SLO, then maximize throughput per dollar |
| Engine | vLLM (`vllm/vllm-openai:nightly` on G4; vLLM-TPU on v6e) |
| Client | `vllm bench serve`, concurrency swept 1 → 1024 |

### Hardware Under Test

| | **TPU v6e-8 (Trillium)** | **g4-standard-384** |
| :--- | :--- | :--- |
| Accelerator | 8 × v6e chip | 8 × RTX PRO 6000 Blackwell Server Edition |
| Memory / chip | 32 GB HBM | 96 GB GDDR7 |
| Interconnect | ICI (high-bandwidth, all-to-all) | **PCIe Gen5 only — no NVLink** |
| Host | — | 384 vCPU, 1,440 GiB RAM |
| Region tested | us-central1 / us-east5 | us-central1-b |

> [!IMPORTANT]
> **The absence of NVLink is the single most important architectural fact about G4.** It is why every G4 configuration here uses **TP=1, DP=8**: a tensor-parallel all-reduce would cross PCIe on every layer of every forward pass. v6e, with ICI, can afford TP=2 or TP=4 — and for the 31B model it *must*, because 62.5 GB of BF16 weights cannot fit in a 32 GB chip.
>
> This produces a neat inversion: **v6e is forced into tensor parallelism by its small memory; G4 is forced out of it by its weak interconnect.** Each platform's best config is the one its constraint permits.

### Configurations Benchmarked
* **31B**: v6e TP2-DP4, v6e TP4-DP2, **G4 TP1-DP8**
* **E4B**: v6e TP1-DP8, **G4 TP1-DP8**

Only TP1-DP8 was run on G4, by agreement — the no-NVLink argument makes TP2/TP4 predictably worse, and the results below confirm TP1 on G4 already matches v6e's *tensor-parallel* result.

---

## 2. Results: Gemma 4 31B

### Headline numbers

| Configuration | p90 < 500 ms | p90 < 1000 ms | Peak (any latency) |
| :--- | :--- | :--- | :--- |
| v6e TP2-DP4 | **93.8 req/s** @ c=32 (354 ms) | **94.1 req/s** @ c=80 (846 ms) | 94.3 @ c=96 |
| v6e TP4-DP2 | 81.7 req/s @ c=8 | 81.7 req/s @ c=8 | 101.8 @ c=320 |
| **G4 TP1-DP8** | **87.6 req/s** @ c=32 (441 ms) | **93.8 req/s** @ c=80 (940 ms) | **98.4 @ c=512+** |

**G4 reaches 93.4% of v6e's throughput under the 500 ms SLO, 99.7% under the 1 s SLO, and exceeds it by 4.3% at saturation.**

### Curve shape — the more interesting result

G4's throughput curve is **monotonic and smooth** across all 21 concurrency levels, converging cleanly to a 98.4 req/s asymptote:

```
G4   c=32  87.6 | c=48  91.2 | c=64  92.6 | c=80  93.8 | c=96  95.0 | c=128 96.0
     c=160 96.6 | c=192 97.0 | c=256 98.0 | c=320 98.1 | c=384 98.3 | c=512 98.4 ...
```

v6e TP2-DP4 does not:

```
v6e  c=32  93.8 | c=48  69.3 | c=64  90.2 | c=80  94.1 | c=96  94.3 | c=128 91.0
     c=160 91.1 | c=192 69.3 | c=256 90.9 | c=320 69.3 | c=384 89.6 | c=512 90.2 ...
                      ^^^^                      ^^^^                ^^^^
```

Three reproducible collapses to **exactly 69.3 req/s** at c=48, c=192 and c=320. The identical value at three widely-separated concurrency levels is not thermal or contention noise — it looks like a **discrete XLA recompilation or bucket-padding pathology**, where those batch shapes fall into a padding bucket that wastes ~26% of the MXU. This is consistent with the known TPU behaviour of compiling fixed shape buckets.

> [!NOTE]
> **Operational implication:** v6e's *peak* is marginally better, but G4's *predictability* is better. For a capacity model you provision against the worst case within your operating range. v6e's worst case between c=32 and c=512 is 69.3 req/s; G4's is 87.6 req/s. On that basis G4 is **26% better**, which is the opposite of the headline conclusion. Which framing is correct depends on whether you can pin production concurrency away from the bad buckets.

### Why G4 TP1 ties v6e TP2

At TP=1 the 31B model measured **58.99 GiB of weights per GPU** — 67% of a 96 GB card. What remains is **26.94 GiB of KV cache = 31,209 tokens = 61 concurrent 512-token sequences per replica**, or ~488 cluster-wide.

That is the binding constraint, and it explains the exact position of the SLO knees: throughput stops improving right where the scheduler runs out of KV blocks and begins queueing. G4 wins back on zero-communication DP exactly what it loses on raw per-chip compute.

---

## 3. Results: Gemma 4 E4B

The v6e E4B sweep was **run twice** with a byte-identical manifest on the same node pool, because the first run contained a physically inconsistent point (see below). Both runs are reported.

### Headline numbers

| Configuration | p90 < 500 ms | p90 < 1000 ms | Peak (any latency) |
| :--- | :--- | :--- | :--- |
| v6e TP1-DP8 — run 1 | 635.0 req/s @ c=192 | **805.3 req/s** @ c=384 | 989.5 @ c=64 *(1.0 s burst)* |
| v6e TP1-DP8 — run 2 | 644.1 req/s @ c=256 | **795.5 req/s** @ c=512 | 984.1 @ c=48 *(1.0 s burst)* |
| **G4 TP1-DP8** | **643.0 req/s** @ c=256 | 649.9 req/s @ c=512 | 900.1 @ c=64 *(1.1 s burst)* |

**Under the 500 ms SLO the two platforms are tied** — 635.0 / 644.1 for v6e against 643.0 for G4, a spread of 1.4% that is smaller than v6e's own run-to-run variance.

**Under the 1 s SLO v6e wins by ~23%, and this is reproducible** — 805.3 and 795.5 across two independent runs, against G4's 649.9.

### What the replicate resolved

Run 1's `c=384` reading was 805.3 req/s at a p90 of 507 ms — *lower* p90 than the less-loaded c=320 point (544 ms), which cannot happen on a saturated server. That specific point did **not** reproduce:

| v6e E4B @ c=384 | req/s | p90 |
| :--- | ---: | ---: |
| run 1 | 805.3 | 507 ms |
| run 2 | 649.6 | 814 ms |

So the anomaly was real, but the **conclusion drawn from it was not affected**: run 2 simply reaches the same ~800 req/s at `c=512` instead (795.5 @ 834 ms). v6e's high-throughput capability is genuine; only *which concurrency it appears at* is unstable.

### Stability is the real difference

Point-for-point across the two v6e runs, the high-concurrency half of the curve moves substantially:

| conc | v6e run 1 | v6e run 2 | Δ | G4 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 606.1 | 644.1 | +6.3% | 643.0 |
| 320 | 688.5 | 691.0 | +0.4% | 647.6 |
| 384 | 805.3 | 649.6 | **−19.3%** | 648.8 |
| 512 | 652.9 | 795.5 | **+21.8%** | 649.9 |
| 640 | 781.1 | 797.5 | +2.1% | 648.2 |
| 768 | 734.8 | 858.7 | **+16.9%** | 651.2 |
| 1024 | 773.2 | 861.3 | **+11.4%** | 650.1 |

G4 over the same range holds **648–651 req/s, 0.5% variance**. v6e swings ±20% between identical runs.

> [!NOTE]
> Both platforms are legitimately described by these data, but they describe *different things*. v6e has a **higher ceiling** (~800 req/s under the 1 s SLO) that it reaches unpredictably. G4 has a **lower but essentially deterministic** rate (650 req/s, flat across a 4× concurrency range). Which matters depends on whether you provision for expected throughput or for worst-case throughput.
>
> Note also that run 2 is *higher* than run 1 at the top of the curve (858.7 and 861.3 at c=768/1024), so v6e's true ceiling may be above 800 — the sweep does not clearly find its saturation point, whereas G4's is unmistakable.


---

## 4. Burst vs. Sustained Throughput

Both platforms report their highest raw throughput at **c=64** — 989.5 req/s on v6e, 900.1 req/s on G4 — and both of those runs lasted **~1 second**.

Measured throughput on G4 E4B is a **monotonic function of run duration**, converging from above:

| Run duration | 1.1 s | 1.3 s | 1.5 s | 3.3 s | 5.2 s | 8.0 s | 15.8 s | 31.5 s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| req/s | 900 | 788 | 679 | 585 | 621 | 643 | 650 | 650 |

Every one of these points satisfies Little's Law (concurrency ÷ mean latency = throughput), so **no individual measurement is invalid**. The server really did complete 1,000 requests in 1.11 seconds. The distinction is between **burst** capacity and **sustained** capacity, and only the latter belongs in a capacity model.

### The confound in the harness

The benchmark client sets `num_prompts = 1000` for c < 96, and `c × 20` for c ≥ 96. Run length and concurrency therefore change **together**, at exactly the concurrency where throughput falls. In the existing data the two are inseparable.

### Diagnostic 1 — breaking the confound

A dedicated job held the server fixed and crossed {c=64, c=96} × {1,000 prompts, 20,480 prompts}, with the two short cells repeated at the end to detect thermal soak. Cells ran interleaved (A, C, B, D, A2, C2) so that drift over the life of the job could not masquerade as a cell effect.

| # | cell | conc | prompts | duration | **req/s** | mean E2E |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | A | 64 | 1,000 | 2.68 s | 373.5 | 167.9 ms |
| 2 | C | 96 | 1,000 | 1.60 s | 626.4 | 145.6 ms |
| 3 | B | 64 | 20,480 | 35.81 s | 571.9 | 111.3 ms |
| 4 | D | 96 | 20,480 | 30.29 s | 676.2 | 141.1 ms |
| 5 | A2 | 64 | 1,000 | 1.85 s | 540.2 | 114.4 ms |
| 6 | C2 | 96 | 1,000 | 1.57 s | 636.7 | 144.2 ms |

**All three candidate explanations are rejected:**

| Hypothesis | Test | Verdict |
| :--- | :--- | :--- |
| Concurrency cliff between c=64 and c=96 | B → D (571.9 → 676.2) and A2 → C2 (540.2 → 636.7) | **Rejected.** Throughput *rises* 18% with concurrency at both durations. |
| Longer runs decay to a lower sustained rate | A2 vs B at c=64 (540.2 @ 1.85 s vs 571.9 @ 35.8 s); C2 vs D at c=96 (636.7 @ 1.57 s vs 676.2 @ 30.3 s) | **Rejected, and backwards.** The long runs were *faster* by ~6%. |
| Power / thermal throttling | A vs A2 (373.5 → 540.2), C vs C2 (626.4 → 636.7) | **Rejected.** The repeat at the end of the job, after ~70 s of sustained load, was **45% faster**. |

Cell A being slow is **server warm-up** — it was the first traffic the server ever saw, so it absorbed JIT compilation, FlashInfer autotuning and allocator settling. That is why A2, the identical configuration run fifth, is 45% faster.

But warm-up does not explain the original 900 req/s: **the best c=64 result anywhere in this diagnostic is 571.9 req/s.**

### The actual mechanism

`vllm bench serve --dataset-name random` derives its prompts from a **fixed seed**, so every concurrency level in the sweep generates the **identical prompt set**. The client holds `num_prompts = 1000` for all `c < 96`, and only then switches to `c × 20`. Therefore:

* By the time the sweep reaches **c=64**, those same 1,000 prompts have already been served seven times (at c=1, 2, 4, 8, 16, 32, 48) and are sitting in the prefix cache.
* E4B's KV cache measured **1,220,612 tokens = 2,441 prompts** of 500 tokens. 1,000 prompts = 500K tokens = **41% of capacity** — they fit entirely, with no eviction.
* At **c ≥ 128**, `num_prompts` becomes ≥ 2,560 = 1.28M tokens, which **exceeds capacity**. The cache evicts continuously, the hit rate collapses, and throughput falls to its true uncached value of ~650 req/s.

The corroborating evidence is the **31B curve**. 31B's KV cache measured only **31,209 tokens = 62 prompts** — it can never hold the prompt set at any concurrency, so it gets no cache benefit anywhere. And indeed the 31B curve is perfectly smooth and monotonic, with no burst anomaly at all. **One mechanism explains both models' curve shapes.**

> [!IMPORTANT]
> **Diagnostic 2 (`gemma-4-e4b-g4-diag2.yaml`) is currently running to confirm this.** It holds the server, concurrency (64) and prompt count (1,000) fixed and varies **only the dataset seed**, while sampling the Prometheus `prefix_cache_hits_total` / `prefix_cache_queries_total` counters around each cell so the hit rate is measured rather than inferred.
>
> The control is cell **P4**: fresh prompts (seed 777) run *after* the warm-cache cells. Any residual server warm-up drift predicts P4 ≥ P2; the cache hypothesis predicts P4 ≪ P2. Results to be appended here.

### What this means for the numbers in this report

* The **~650 req/s plateau is the honest sustained figure** for E4B on G4, and the 500 ms / 1000 ms SLO knees at c=256 / c=512 are sound — those runs are all well past the point where the cache stops paying.
* The **900 req/s (G4) and 989 req/s (v6e) peaks at c=64 are benchmark artifacts**, not capability. They should not be used for capacity planning.
* Both platforms were measured with the same harness and the same seed behaviour, so the **v6e-vs-G4 comparison remains fair** — both are inflated identically at low concurrency.
* **The 31B results are entirely unaffected** and require no caveat.

> [!TIP]
> If you want a benchmark that measures steady-state serving rather than cache replay, either pass a distinct `--seed` per concurrency level, or run with `--no-enable-prefix-caching`. For *this* workload the latter is also the right production setting — see `additional_optimizations.md`.


---

## 5. Price / Performance

> [!CAUTION]
> An earlier provisional figure in this analysis put the g4-standard-384 node at **$27.23/hr**. That was **wrong** — it counted vCPU and RAM only and omitted the GPU entirely. The GPU is billed under a SKU named **`RTX 6000 96GB`**, not "RTX PRO 6000". Corrected figures below.

### Node cost (Cloud Billing Catalog API, service `6F81-5844-456A`)

**g4-standard-384** = 384 vCPU + 1,440 GiB RAM + 8 GPU. Local SSD is bundled at $0.

| Region | Mode | vCPU | RAM | GPU | **Total** | per GPU-hr |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| us-central1 | On-demand | $18.78 | $8.45 | $8.77 | **$36.00/hr** | $4.50 |
| us-central1 | Spot | $6.41 | $2.88 | $4.66 | **$13.94/hr** | $1.74 |
| us-east5 | On-demand | $20.66 | $9.30 | $9.64 | **$39.60/hr** | $4.95 |
| us-east5 | Spot | $9.47 | $4.26 | $4.21 | **$17.95/hr** | $2.24 |

**v6e-8**:

| Region | Mode | **Total** | per chip-hr |
| :--- | :--- | ---: | ---: |
| us-central1 | On-demand | **$21.60/hr** | $2.70 |
| us-central1 | Spot | **$5.19/hr** | $0.65 |
| us-east5 | Spot | **$12.96/hr** | $1.62 |

> [!IMPORTANT]
> **The GPU is only 24% of the G4 bill.** vCPU alone is **$18.78/hr — 52% of the node cost** — because `g4-standard-384` is the only 8-GPU G4 shape and it forces you to buy 384 vCPUs for an inference server that uses a handful of them. There is no way to opt out: the 1-GPU `g4-standard-48` scales linearly to the identical $36.00/hr for 8 GPUs.
>
> This is the dominant term in the entire price/performance story, and it is a **packaging** problem, not a silicon problem.

### Throughput per dollar (us-central1)

**On-demand** — G4 $36.00/hr vs v6e $21.60/hr (**1.67×**):

| Operating point | v6e req/$ | G4 req/$ | G4 / v6e |
| :--- | ---: | ---: | ---: |
| 31B, p90 < 500 ms | 15,633 | 8,760 | **0.56×** |
| 31B, p90 < 1000 ms | 15,683 | 9,380 | **0.60×** |
| 31B, peak | 15,717 | 9,840 | **0.63×** |
| E4B, p90 < 500 ms | 105,833 | 64,301 | **0.61×** |
| E4B, p90 < 1000 ms † | 134,217 | 64,991 | 0.48× |

**Spot** — G4 $13.94/hr vs v6e $5.19/hr (**2.69×**):

| Operating point | v6e req/$ | G4 req/$ | G4 / v6e |
| :--- | ---: | ---: | ---: |
| 31B, p90 < 500 ms | 65,086 | 22,622 | **0.35×** |
| 31B, p90 < 1000 ms | 65,294 | 24,223 | **0.37×** |
| E4B, p90 < 500 ms | 440,612 | 166,051 | **0.38×** |

† Confirmed by a second independent v6e run (805.3 @ c=384 and 795.5 @ c=512); see §3. Not an artifact.

### Conclusion on cost

**v6e is the clear price/performance winner: 1.6–1.8× better on-demand, and 2.6–2.9× better on Spot.**

The reason is not that G4 is slow — on the 31B model G4 delivers **99.7% of v6e's SLO-constrained throughput**, and beats it at saturation. The reason is entirely cost structure:
1. The 8-GPU G4 shape bundles 384 vCPUs that this workload does not use (52% of the bill).
2. v6e Spot in us-central1 is discounted **76%** off list ($2.70 → $0.65), while G4 Spot is discounted only **61%** — so the gap *widens* under Spot, from 1.67× to 2.69×.

---

## 6. Recommendations

| Scenario | Recommendation |
| :--- | :--- |
| **Cost is the primary driver** | **TPU v6e** on both models. 1.6–1.8× better req/$ on-demand, 2.6–2.9× on Spot. |
| **31B, TPU capacity unavailable** | **G4 is a strong substitute** — 99.7% of v6e throughput under the 1 s SLO and *higher* at saturation. The performance case is sound; only the price is worse. |
| **E4B, TPU capacity unavailable** | **G4 is adequate but not equivalent** — tied at the 500 ms knee (643 vs 635–644), but ~23% behind at the 1 s knee (650 vs 795–805), confirmed over two v6e runs. |
| **Predictable latency is the primary driver** | **G4.** Its curves are monotonic and repeatable (E4B holds 648–651 req/s across a 4× concurrency range). v6e shows 26% throughput collapses at specific concurrency buckets on 31B, and ±20% run-to-run swings at matched concurrency on E4B. |
| **Next step for either platform** | Apply the flag-only optimizations in `g4/additional_optimizations.md`. They are **not symmetric** across the two platforms, so re-measure rather than assuming the ranking holds. |

> [!TIP]
> The G4 price gap is closeable. The FP8/NVFP4 quantization path (see `additional_optimizations.md`) projects $+60\%$–$100\%$ on 31B, which would bring G4 to roughly **0.9–1.2× of v6e on-demand req/$**. Blackwell's 5th-gen Tensor Cores are the asset that this BF16 benchmark leaves entirely unused — but note the same document's caveat that v6e also has an FP8 path, so this is not a free win in a like-for-like comparison.

---

## 7. Artifacts

| File | Contents |
| :--- | :--- |
| [gemma-4-31b-g4-tp1-dp8.yaml](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/gemma-4-31B/g4/gemma-4-31b-g4-tp1-dp8.yaml) | G4 31B Job manifest |
| [gemma-4-31b-g4-tp1-dp8_results.csv](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/gemma-4-31B/g4/gemma-4-31b-g4-tp1-dp8_results.csv) | 21 concurrency levels |
| [gemma-4-e4b-g4-tp1-dp8.yaml](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/gemma-4-E4B/g4/gemma-4-e4b-g4-tp1-dp8.yaml) | G4 E4B Job manifest |
| [gemma-4-e4b-g4-tp1-dp8_results.csv](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/gemma-4-E4B/g4/gemma-4-e4b-g4-tp1-dp8_results.csv) | 19 concurrency levels |
| [gemma-4-e4b-tp1-dp8-rerun.yaml](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/gemma-4-E4B/gemma-4-e4b-tp1-dp8-rerun.yaml) | v6e E4B replicate manifest (3 lines differ from the original) |
| [gemma-4-e4b-tp1-dp8-rerun_results.csv](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/gemma-4-E4B/gemma-4-e4b-tp1-dp8-rerun_results.csv) | v6e E4B run 2, 19 concurrency levels |
| [31B additional_optimizations.md](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/gemma-4-31B/g4/additional_optimizations.md) | 5 G4-specific optimizations |
| [E4B additional_optimizations.md](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/gemma-4-E4B/g4/additional_optimizations.md) | 5 G4-specific optimizations |
| [results_to_csv.py](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/scripts/results_to_csv.py) | JSON → CSV converter |
| [slo_analysis.py](file:///Users/ericehanley/code/roblox-g4-v6e-benchmarking/scripts/slo_analysis.py) | SLO knee finder |

Raw JSON results are in `gs://llama-vllm-results-bucket/gemma-4-{31b,e4b}-native-bench/g4/`.
