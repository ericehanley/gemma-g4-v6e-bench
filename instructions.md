# Gemma 4 E4B & 31B Benchmarking

## General Benchmarking Features

The workload we are aiming for should be representative of the binary classification workload that Roblox currently runs. Assume the workload is 500 input tokens with 50 fixed prefix tokens. Output is 1 token
for binary prediction. We want to maximize throughput with p90 E2E latency less than 1 second and we also want to see where we hit 500 ms p90 E2E latency. The overall goal is to maximize price/performance which generally means the less chips we have to use the better.

## Directions

1. Develop four total manifests based on gemma-4-e4b-tp1-dp8.yaml: an updated gemma-4-e4b-tp1-dp8.yaml, a new gemma-4-e4b-tp2-dp4.yaml, a new gemma-4-e4b-tp4-dp2.yaml, a new gemma-4-e4b-tp8-dp1.yaml
  a. For the workload described above, determine the optimal vLLM server settings including gpu-memory-utilization (maximize without OOM), max-model-len (minimize to what is needed for the workload), max-num-batched-tokens, and max-num-seqs. Research these variables and compare to the workload to come up with the best configuration - do not use an auto-tuning tool or anything like that.
  b. Review the following web page to understand the TPU environment variables and then set them as appropriate for the workload. If it is unclear, ask me questions to determine the best course of action: https://vllm--2806.org.readthedocs.build/projects/tpu/en/2806/environment_variables/.
  c. Use the vllm/vllm-tpu:nightly for the server and the client containers.
  d. For the client container, update the workload to reflect the workload described above. We still want to understand performance at different concurrency levels. Keep in mind the goals are to maximize throughput at the stated p90 latency constraints. So the concurrency levels you test should allow you to see where we approach and exceed that constraint.
  e. After making the above manifests, propose at least three additional ways we could optimize this inference workload while still using the vLLM engine.
