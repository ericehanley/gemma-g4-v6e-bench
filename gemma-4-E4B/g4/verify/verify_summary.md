# Verify run summary (gemma-4-E4B/g4/verify/results)

*Kept from the pre-replay work (2026-10-06) because both customer-replay results reports cite it: the one-GPU fast-prefill A/B on the synthetic 50 + 450-token spec (A_baseline 82.3 vs B_fastprefill 136.8 req/s at c = 64; 73.2 vs 108.2 at c = 8). The raw `results/` and the per-variant scripts were archived outside git on 2026-10-09; only this summary and the Job manifest remain.*

```
image vllm: 0.30.1rc1.dev709+g21d93d0d8
torch                                    2.13.0+cu130
transformers                             5.18.0
vllm                                     0.30.1rc1.dev709+g21d93d0d8
```

### A_baseline / phase1_clean

| c | N | seed | req/s | p50 ms | p90 ms | p99 ms | TTFT p90 | hit % | tok/req | SM MHz | W | util % | failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1000 | 1001 | 44.8 | 22.3 | 22.5 | 23.1 | 22.5 | 6.7 | 500.0 | 2376 | 416 | 80 | 0 |
| 2 | 1000 | 1002 | 52.1 | 38.4 | 38.6 | 39.4 | 38.6 | 7.0 | 500.0 | 2371 | 458 | 88 | 0 |
| 4 | 1000 | 1004 | 60.6 | 65.9 | 66.2 | 66.6 | 66.2 | 6.5 | 500.0 | 2366 | 520 | 93 | 0 |
| 8 | 1000 | 1008 | 73.2 | 109.1 | 109.6 | 111.0 | 109.6 | 6.5 | 500.0 | 2366 | 551 | 89 | 0 |
| 16 | 1000 | 1016 | 78.3 | 204.1 | 205.1 | 205.9 | 205.1 | 6.5 | 500.0 | 2338 | 565 | 94 | 0 |
| 24 | 1000 | 1024 | 81.4 | 294.7 | 296.1 | 297.1 | 296.1 | 6.8 | 500.0 | 2310 | 560 | 92 | 0 |
| 32 | 1000 | 1032 | 81.2 | 393.2 | 395.3 | 427.5 | 395.3 | 6.6 | 500.0 | 2306 | 566 | 93 | 0 |
| 40 | 1000 | 1040 | 81.6 | 490.0 | 492.6 | 498.4 | 492.6 | 6.7 | 500.0 | 2311 | 538 | 92 | 0 |
| 48 | 1000 | 1048 | 81.3 | 589.3 | 592.2 | 618.9 | 592.2 | 6.6 | 500.0 | 2304 | 554 | 92 | 0 |
| 64 | 1280 | 1064 | 82.3 | 849.7 | 856.1 | 860.9 | 856.1 | 7.0 | 500.0 | 2294 | 556 | 94 | 0 |
| 80 | 1600 | 1080 | 82.1 | 1058.8 | 1070.1 | 1072.4 | 1070.1 | 6.7 | 500.0 | 2293 | 567 | 95 | 0 |
| 96 | 1920 | 1096 | 82.1 | 1073.0 | 1284.7 | 1286.9 | 1284.7 | 6.8 | 500.0 | 2286 | 586 | 100 | 0 |
| 128 | 2560 | 1128 | 82.2 | 1500.2 | 1712.2 | 1715.8 | 1712.2 | 7.0 | 500.0 | 2290 | 571 | 96 | 0 |
| 192 | 3840 | 1192 | 82.2 | 2354.5 | 2357.6 | 2358.9 | 2357.6 | 6.9 | 500.0 | 2282 | 593 | 100 | 0 |
| 256 | 5120 | 1256 | 82.2 | 3172.9 | 3212.3 | 3216.9 | 3212.3 | 7.0 | 500.0 | 2284 | 589 | 98 | 0 |

### A_baseline / phase2_artifact

| c | N | seed | req/s | p50 ms | p90 ms | p99 ms | TTFT p90 | hit % | tok/req | SM MHz | W | util % | failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1000 | 0 | 44.6 | 22.4 | 22.7 | 23.2 | 22.7 | 6.6 | 500.0 | 2379 | 426 | 79 | 0 |
| 2 | 1000 | 0 | 86.8 | 22.9 | 23.1 | 23.5 | 23.1 | 96.0 | 500.0 | 2422 | 277 | 88 | 0 |
| 4 | 1000 | 0 | 129.6 | 30.8 | 31.0 | 31.5 | 31.0 | 96.0 | 500.0 | 2422 | 292 | 99 | 0 |
| 8 | 1000 | 0 | 253.9 | 31.5 | 31.9 | 32.9 | 31.9 | 96.0 | 500.0 | 2421 | 242 | 76 | 0 |
| 16 | 1000 | 0 | 433.2 | 34.6 | 43.8 | 45.1 | 43.8 | 96.0 | 500.0 | 2412 | 258 | 66 | 0 |
| 32 | 1000 | 0 | 666.3 | 46.3 | 52.6 | 71.9 | 52.6 | 96.0 | 500.0 | 2400 | 280 | 50 | 0 |
| 48 | 1000 | 0 | 695.0 | 68.9 | 77.2 | 103.9 | 77.2 | 96.0 | 500.0 | 2392 | 303 | 99 | 0 |
| 64 | 1000 | 0 | 700.0 | 90.0 | 97.0 | 135.2 | 97.0 | 96.0 | 500.0 | 2404 | 258 | 50 | 0 |
| 96 | 1000 | 0 | 702.4 | 127.6 | 145.3 | 188.3 | 145.3 | 96.0 | 500.0 | 2378 | 296 | 98 | 0 |
| 128 | 1000 | 0 | 684.4 | 178.2 | 190.3 | 245.4 | 190.3 | 96.0 | 500.0 | 2378 | 302 | 99 | 0 |

### B_fastprefill / phase1_clean

| c | N | seed | req/s | p50 ms | p90 ms | p99 ms | TTFT p90 | hit % | tok/req | SM MHz | W | util % | failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1000 | 1001 | 52.8 | 18.9 | 19.1 | 19.9 | 19.1 | 6.7 | 500.0 | 2389 | 332 | 68 | 0 |
| 2 | 1000 | 1002 | 63.1 | 31.6 | 31.9 | 32.2 | 31.9 | 7.0 | 500.0 | 2387 | 389 | 84 | 0 |
| 4 | 1000 | 1004 | 89.2 | 44.7 | 45.1 | 46.5 | 45.1 | 6.5 | 500.0 | 2378 | 428 | 83 | 0 |
| 8 | 1000 | 1008 | 108.2 | 73.7 | 74.4 | 76.5 | 74.4 | 6.5 | 500.0 | 2382 | 463 | 84 | 0 |
| 16 | 1000 | 1016 | 120.4 | 132.2 | 133.0 | 145.5 | 133.0 | 6.5 | 500.0 | 2377 | 516 | 88 | 0 |
| 24 | 1000 | 1024 | 130.3 | 183.6 | 184.6 | 209.3 | 184.6 | 6.8 | 500.0 | 2376 | 539 | 92 | 0 |
| 32 | 1000 | 1032 | 131.3 | 242.4 | 243.8 | 245.0 | 243.8 | 6.6 | 500.0 | 2362 | 538 | 87 | 0 |
| 40 | 1000 | 1040 | 135.0 | 295.1 | 296.7 | 337.1 | 296.7 | 6.7 | 500.0 | 2348 | 541 | 98 | 0 |
| 48 | 1000 | 1048 | 134.9 | 354.3 | 355.7 | 357.7 | 355.7 | 6.6 | 500.0 | 2346 | 560 | 99 | 0 |
| 64 | 1280 | 1064 | 136.8 | 509.7 | 513.5 | 521.2 | 513.5 | 7.0 | 500.0 | 2337 | 558 | 99 | 0 |
| 80 | 1600 | 1080 | 136.2 | 637.6 | 643.1 | 644.8 | 643.1 | 6.7 | 500.0 | 2326 | 578 | 99 | 0 |
| 96 | 1920 | 1096 | 135.9 | 646.6 | 773.4 | 776.5 | 773.4 | 6.8 | 500.0 | 2324 | 560 | 93 | 0 |
| 128 | 2560 | 1128 | 135.8 | 906.8 | 1033.0 | 1038.2 | 1033.0 | 7.0 | 500.0 | 2315 | 571 | 94 | 0 |
| 192 | 3840 | 1192 | 135.6 | 1422.1 | 1428.7 | 1431.0 | 1428.7 | 6.9 | 500.0 | 2310 | 586 | 96 | 0 |
| 256 | 5120 | 1256 | 135.6 | 1918.7 | 1946.8 | 1949.9 | 1946.8 | 7.0 | 500.0 | 2311 | 573 | 94 | 0 |

### C_fp8 / phase1_clean

| c | N | seed | req/s | p50 ms | p90 ms | p99 ms | TTFT p90 | hit % | tok/req | SM MHz | W | util % | failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1000 | 1001 | 52.1 | 19.1 | 19.5 | 20.4 | 19.5 | 6.7 | 500.0 | 2389 | 326 | 74 | 0 |
| 2 | 1000 | 1002 | 62.9 | 31.8 | 32.0 | 32.6 | 32.0 | 7.0 | 500.0 | 2384 | 359 | 86 | 0 |
| 4 | 1000 | 1004 | 80.1 | 49.9 | 50.1 | 50.6 | 50.1 | 6.5 | 500.0 | 2363 | 435 | 98 | 0 |
| 8 | 1000 | 1008 | 107.7 | 74.0 | 74.7 | 79.4 | 74.7 | 6.5 | 500.0 | 2351 | 500 | 86 | 0 |
| 16 | 1000 | 1016 | 117.7 | 135.6 | 136.3 | 137.1 | 136.3 | 6.5 | 500.0 | 2352 | 505 | 88 | 0 |
| 24 | 1000 | 1024 | 123.1 | 194.7 | 195.8 | 219.7 | 195.8 | 6.8 | 500.0 | 2350 | 531 | 85 | 0 |
| 32 | 1000 | 1032 | 125.9 | 253.1 | 254.5 | 271.6 | 254.5 | 6.6 | 500.0 | 2338 | 516 | 89 | 0 |
| 40 | 1000 | 1040 | 128.0 | 311.6 | 313.0 | 350.3 | 313.0 | 6.7 | 500.0 | 2331 | 532 | 77 | 0 |
| 48 | 1000 | 1048 | 128.2 | 372.7 | 374.1 | 386.2 | 374.1 | 6.6 | 500.0 | 2321 | 531 | 89 | 0 |
| 64 | 1280 | 1064 | 130.2 | 535.8 | 539.1 | 540.6 | 539.1 | 7.0 | 500.0 | 2317 | 517 | 91 | 0 |
| 80 | 1600 | 1080 | 129.7 | 668.5 | 675.5 | 677.3 | 675.5 | 6.7 | 500.0 | 2300 | 567 | 98 | 0 |
| 96 | 1920 | 1096 | 129.5 | 679.9 | 812.2 | 815.8 | 812.2 | 6.8 | 500.0 | 2295 | 561 | 93 | 0 |
| 128 | 2560 | 1128 | 129.4 | 951.5 | 1084.8 | 1088.5 | 1084.8 | 7.0 | 500.0 | 2289 | 556 | 93 | 0 |
| 192 | 3840 | 1192 | 129.5 | 1491.0 | 1497.3 | 1498.8 | 1497.3 | 7.0 | 500.0 | 2280 | 574 | 97 | 0 |
| 256 | 5120 | 1256 | 129.4 | 1926.7 | 2040.5 | 2044.5 | 2040.5 | 7.1 | 500.0 | 2276 | 585 | 98 | 0 |

### D_fastprefill_fp8 / phase1_clean

| c | N | seed | req/s | p50 ms | p90 ms | p99 ms | TTFT p90 | hit % | tok/req | SM MHz | W | util % | failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1000 | 1001 | 62.3 | 16.0 | 16.2 | 16.8 | 16.2 | 6.7 | 500.0 | 2398 | 271 | 67 | 0 |
| 2 | 1000 | 1002 | 77.9 | 25.6 | 25.8 | 26.8 | 25.8 | 7.0 | 500.0 | 2396 | 319 | 81 | 0 |
| 4 | 1000 | 1004 | 107.1 | 38.1 | 38.4 | 39.0 | 38.4 | 6.5 | 500.0 | 2377 | 366 | 86 | 0 |
| 8 | 1000 | 1008 | 154.2 | 51.6 | 52.1 | 53.8 | 52.1 | 6.5 | 500.0 | 2365 | 420 | 85 | 0 |
| 16 | 1000 | 1016 | 180.0 | 88.4 | 89.0 | 89.7 | 89.0 | 6.5 | 500.0 | 2367 | 473 | 98 | 0 |
| 24 | 1000 | 1024 | 195.7 | 122.1 | 122.8 | 147.9 | 122.8 | 6.8 | 500.0 | 2362 | 487 | 82 | 0 |
| 32 | 1000 | 1032 | 202.6 | 156.7 | 157.5 | 172.4 | 157.5 | 6.6 | 500.0 | 2365 | 482 | 82 | 0 |
| 40 | 1000 | 1040 | 207.6 | 191.1 | 192.1 | 233.3 | 192.1 | 6.7 | 500.0 | 2376 | 443 | 82 | 0 |
| 48 | 1000 | 1048 | 210.0 | 226.3 | 227.5 | 234.2 | 227.5 | 6.6 | 500.0 | 2362 | 490 | 66 | 0 |
| 64 | 1280 | 1064 | 214.1 | 324.1 | 326.7 | 358.5 | 326.7 | 7.0 | 500.0 | 2349 | 486 | 85 | 0 |
| 80 | 1600 | 1080 | 213.7 | 404.0 | 408.3 | 414.9 | 408.3 | 6.7 | 500.0 | 2344 | 544 | 99 | 0 |
| 96 | 1920 | 1096 | 213.5 | 410.8 | 490.3 | 498.9 | 490.3 | 6.8 | 500.0 | 2343 | 532 | 89 | 0 |
| 128 | 2560 | 1128 | 213.5 | 574.3 | 654.5 | 657.1 | 654.5 | 7.0 | 500.0 | 2334 | 552 | 91 | 0 |
| 192 | 3840 | 1192 | 213.0 | 902.8 | 907.9 | 910.0 | 907.9 | 7.0 | 500.0 | 2324 | 571 | 94 | 0 |
| 256 | 5120 | 1256 | 212.2 | 1201.1 | 1241.1 | 1245.0 | 1241.1 | 7.1 | 500.0 | 2314 | 579 | 95 | 0 |

## Knees (1 GPU; x8 = naive g4-standard-384 extrapolation)

| variant / phase | best req/s, p90<500ms | best req/s, p90<1s | peak req/s | x8 of p90<1s |
| :--- | ---: | ---: | ---: | ---: |
| A_baseline / phase1_clean | 81.6 @ c=40 (p90 493) | 82.3 @ c=64 (p90 856) | 82.3 @ c=64 (p90 856) | 658 |
| A_baseline / phase2_artifact | 702.4 @ c=96 (p90 145) | 702.4 @ c=96 (p90 145) | 702.4 @ c=96 (p90 145) | 5619 |
| B_fastprefill / phase1_clean | 135.0 @ c=40 (p90 297) | 136.8 @ c=64 (p90 513) | 136.8 @ c=64 (p90 513) | 1094 |
| C_fp8 / phase1_clean | 128.2 @ c=48 (p90 374) | 130.2 @ c=64 (p90 539) | 130.2 @ c=64 (p90 539) | 1042 |
| D_fastprefill_fp8 / phase1_clean | 214.1 @ c=64 (p90 327) | 214.1 @ c=64 (p90 327) | 214.1 @ c=64 (p90 327) | 1712 |

## Parity vs A_baseline (temperature 0, 1 output token, 200 prompts)

| variant | compared | top-1 agree | mean abs dlogprob (A's top-1) | max abs dlogprob | A top-1 missing from top-5 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| B_fastprefill | 200 | 196/200 (98.0%) | 0.0323 | 0.1256 | 0 |
| C_fp8 | 200 | 148/200 (74.0%) | 0.3560 | 1.3762 | 0 |
| D_fastprefill_fp8 | 200 | 158/200 (79.0%) | 0.3465 | 1.0646 | 0 |

A_baseline top-1 outputs: [(' UNS', 129), ('\n\n', 40), (' SAFE', 30), (' [', 1)]

## Server-log evidence


### A_baseline (36642 lines)

```
[kv pool] (EngineCore pid=349) INFO 10-06 15:12:17 [kv_cache_utils.py:2481] GPU KV cache size: 1,219,986 tokens, Maximum concurrency for 512 tokens per request: 2382.79x
[attention] (APIServer pid=69) INFO 10-06 15:09:40 [config.py:266] Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.
[attention] (EngineCore pid=349) INFO 10-06 15:10:08 [cuda.py:466] Using AttentionBackendEnum.TRITON_ATTN backend.
[quant] (EngineCore pid=349) INFO 10-06 15:10:05 [core.py:129] Initializing a V1 LLM engine (v0.30.1rc1.dev709+g21d93d0d8) with config: model='google/gemma-4-E4B', speculative_config=None, tokenizer='google/gemma-4-E4B', skip_tokenizer_init=False, tokenizer_mode=auto,
[startup] (EngineCore pid=349) INFO 10-06 15:11:03 [model_runner.py:407] Model loading took 15.2 GiB memory and 56.281756 seconds
[startup] (EngineCore pid=349) INFO 10-06 15:12:16 [model_runner.py:1103] Graph capturing finished in 8 secs, took 0.43 GiB
[startup] (EngineCore pid=349) INFO 10-06 15:12:17 [gpu_worker.py:725] Available KV cache memory: 68.06 GiB
[startup] (EngineCore pid=349) INFO 10-06 15:12:23 [model_runner.py:1103] Graph capturing finished in 4 secs, took 0.23 GiB
[startup] (EngineCore pid=349) INFO 10-06 15:12:24 [core.py:373] init engine (profile, create kv cache, warmup model) took 80.30 s (compilation: 35.57 s)
```

### B_fastprefill (26546 lines)

```
[kv pool] (EngineCore pid=6541) INFO 10-06 15:30:48 [kv_cache_utils.py:2481] GPU KV cache size: 1,214,844 tokens, Maximum concurrency for 512 tokens per request: 2372.74x
[attention] (APIServer pid=6403) INFO 10-06 15:28:14 [config.py:266] Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.
[attention] (EngineCore pid=6541) INFO 10-06 15:28:50 [cuda.py:466] Using AttentionBackendEnum.TRITON_ATTN backend.
[quant] (EngineCore pid=6541) INFO 10-06 15:28:47 [core.py:129] Initializing a V1 LLM engine (v0.30.1rc1.dev709+g21d93d0d8) with config: model='google/gemma-4-E4B', speculative_config=None, tokenizer='google/gemma-4-E4B', skip_tokenizer_init=False, tokenizer_mode=auto
[fast-prefill] (APIServer pid=6403) INFO 10-06 15:28:03 [api_utils.py:286] non-default args: {'model_tag': 'google/gemma-4-E4B', 'host': '0.0.0.0', 'model': 'google/gemma-4-E4B', 'trust_remote_code': True, 'dtype': 'bfloat16', 'max_model_len': 512, 'gpu_memory_utilization': 
[fast-prefill] (APIServer pid=6403) WARNING 10-06 15:28:15 [vllm.py:2051] --kv-sharing-fast-prefill requires changes on model side for correctness and to realize prefill savings.
[fast-prefill] (EngineCore pid=6541) WARNING 10-06 15:31:04 [vllm.py:2051] --kv-sharing-fast-prefill requires changes on model side for correctness and to realize prefill savings.
[startup] (EngineCore pid=6541) INFO 10-06 15:29:32 [model_runner.py:407] Model loading took 15.4 GiB memory and 42.628810 seconds
[startup] (EngineCore pid=6541) INFO 10-06 15:30:47 [model_runner.py:1103] Graph capturing finished in 9 secs, took 0.44 GiB
[startup] (EngineCore pid=6541) INFO 10-06 15:30:48 [gpu_worker.py:725] Available KV cache memory: 67.78 GiB
[startup] (EngineCore pid=6541) INFO 10-06 15:30:54 [model_runner.py:1103] Graph capturing finished in 4 secs, took 0.25 GiB
[startup] (EngineCore pid=6541) INFO 10-06 15:30:55 [core.py:373] init engine (profile, create kv cache, warmup model) took 82.80 s (compilation: 45.21 s)
```

### C_fp8 (26541 lines)

```
[kv pool] (EngineCore pid=10769) INFO 10-06 15:43:21 [kv_cache_utils.py:2481] GPU KV cache size: 1,293,854 tokens, Maximum concurrency for 512 tokens per request: 2527.06x
[attention] (APIServer pid=10524) INFO 10-06 15:40:45 [config.py:266] Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.
[attention] (EngineCore pid=10769) INFO 10-06 15:41:21 [cuda.py:466] Using AttentionBackendEnum.TRITON_ATTN backend.
[quant] (APIServer pid=10524) INFO 10-06 15:40:34 [api_utils.py:286] non-default args: {'model_tag': 'google/gemma-4-E4B', 'host': '0.0.0.0', 'model': 'google/gemma-4-E4B', 'trust_remote_code': True, 'dtype': 'bfloat16', 'max_model_len': 512, 'quantization': 'fp8', 'g
[quant] (APIServer pid=10524) WARNING 10-06 15:40:45 [weight_utils.py:343] --quantization fp8 is deprecated for online quantization; use --quantization fp8_per_tensor instead.
[quant] (EngineCore pid=10769) INFO 10-06 15:41:19 [core.py:129] Initializing a V1 LLM engine (v0.30.1rc1.dev709+g21d93d0d8) with config: model='google/gemma-4-E4B', speculative_config=None, tokenizer='google/gemma-4-E4B', skip_tokenizer_init=False, tokenizer_mode=aut
[quant] (EngineCore pid=10769) INFO 10-06 15:41:21 [__init__.py:758] Selected CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod
[quant] (EngineCore pid=10769) INFO 10-06 15:41:22 [base_loader.py:133] Quantized 502 layers of types: audio_tower.output_proj: 1 (from linear: fp8_per_tensor_static); audio_tower.subsample_conv_projection.input_proj_linear: 1 (from linear: fp8_per_tensor_static); emb
[startup] (EngineCore pid=10769) INFO 10-06 15:41:53 [model_runner.py:407] Model loading took 11.13 GiB memory and 33.137943 seconds
[startup] (EngineCore pid=10769) INFO 10-06 15:43:20 [model_runner.py:1103] Graph capturing finished in 9 secs, took 0.39 GiB
[startup] (EngineCore pid=10769) INFO 10-06 15:43:21 [gpu_worker.py:725] Available KV cache memory: 72.18 GiB
[startup] (EngineCore pid=10769) INFO 10-06 15:43:27 [model_runner.py:1103] Graph capturing finished in 4 secs, took 0.19 GiB
[startup] (EngineCore pid=10769) INFO 10-06 15:43:28 [core.py:373] init engine (profile, create kv cache, warmup model) took 94.30 s (compilation: 55.68 s)
```

### D_fastprefill_fp8 (26545 lines)

```
[kv pool] (EngineCore pid=15007) INFO 10-06 15:56:17 [kv_cache_utils.py:2481] GPU KV cache size: 1,288,887 tokens, Maximum concurrency for 512 tokens per request: 2517.36x
[attention] (APIServer pid=14756) INFO 10-06 15:53:24 [config.py:266] Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.
[attention] (EngineCore pid=15007) INFO 10-06 15:54:00 [cuda.py:466] Using AttentionBackendEnum.TRITON_ATTN backend.
[quant] (APIServer pid=14756) INFO 10-06 15:53:14 [api_utils.py:286] non-default args: {'model_tag': 'google/gemma-4-E4B', 'host': '0.0.0.0', 'model': 'google/gemma-4-E4B', 'trust_remote_code': True, 'dtype': 'bfloat16', 'max_model_len': 512, 'quantization': 'fp8', 'g
[quant] (APIServer pid=14756) WARNING 10-06 15:53:25 [weight_utils.py:343] --quantization fp8 is deprecated for online quantization; use --quantization fp8_per_tensor instead.
[quant] (EngineCore pid=15007) INFO 10-06 15:53:59 [core.py:129] Initializing a V1 LLM engine (v0.30.1rc1.dev709+g21d93d0d8) with config: model='google/gemma-4-E4B', speculative_config=None, tokenizer='google/gemma-4-E4B', skip_tokenizer_init=False, tokenizer_mode=aut
[quant] (EngineCore pid=15007) INFO 10-06 15:54:00 [__init__.py:758] Selected CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod
[quant] (EngineCore pid=15007) INFO 10-06 15:54:01 [base_loader.py:133] Quantized 502 layers of types: audio_tower.output_proj: 1 (from linear: fp8_per_tensor_static); audio_tower.subsample_conv_projection.input_proj_linear: 1 (from linear: fp8_per_tensor_static); emb
[fast-prefill] (APIServer pid=14756) INFO 10-06 15:53:14 [api_utils.py:286] non-default args: {'model_tag': 'google/gemma-4-E4B', 'host': '0.0.0.0', 'model': 'google/gemma-4-E4B', 'trust_remote_code': True, 'dtype': 'bfloat16', 'max_model_len': 512, 'quantization': 'fp8', 'g
[fast-prefill] (APIServer pid=14756) WARNING 10-06 15:53:25 [vllm.py:2051] --kv-sharing-fast-prefill requires changes on model side for correctness and to realize prefill savings.
[fast-prefill] (EngineCore pid=15007) WARNING 10-06 15:56:33 [vllm.py:2051] --kv-sharing-fast-prefill requires changes on model side for correctness and to realize prefill savings.
[startup] (EngineCore pid=15007) INFO 10-06 15:54:33 [model_runner.py:407] Model loading took 11.33 GiB memory and 33.083583 seconds
[startup] (EngineCore pid=15007) INFO 10-06 15:56:16 [model_runner.py:1103] Graph capturing finished in 9 secs, took 0.42 GiB
[startup] (EngineCore pid=15007) INFO 10-06 15:56:17 [gpu_worker.py:725] Available KV cache memory: 71.91 GiB
[startup] (EngineCore pid=15007) INFO 10-06 15:56:23 [model_runner.py:1103] Graph capturing finished in 4 secs, took 0.21 GiB
[startup] (EngineCore pid=15007) INFO 10-06 15:56:24 [core.py:373] init engine (profile, create kv cache, warmup model) took 111.44 s (compilation: 73.08 s)
```
