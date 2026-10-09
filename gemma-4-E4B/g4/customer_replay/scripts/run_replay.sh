#!/usr/bin/env bash
# Customer replay: safechat-100k against vLLM v0.30.0 with the customer's exact server flags,
# on ONE RTX PRO 6000 (g4-standard-48), the machine shape they measured on.
#
# Goal: see whether one GPU gives their 190-240 req/s at p50/p99 = 70/110 ms.
#
# Sequence (details in rbx_loadgen.py):
#   0  server start (customer flags + stand-in middleware), GPU/CPU telemetry
#   1  wait for the dataset to be streamed into /data (tmpfs), render prompts, length stats
#   2  smoke test (labels come back, p_unsafe sane, prefix-cache hit rate ~27%)
#   3  closed-loop sweep c = 1 ... 256, 60 s each
#   4  open-loop Poisson 120 ... 320 req/s, 120 s each, then +80 until p50 > 150 ms
#   5  10-min soak at the rate where p50 ~ 70 ms
#   6  variants at that rate: file order, plain "s1:" format, no allowed_token_ids,
#      /v1/completions, middleware bypassed
#   7  tie-back: `vllm bench serve` on our old synthetic 500-token spec (expect ~136 req/s)
#
# Results stay in ${OUT} (no cloud credentials in the pod). When ${OUT}/DONE appears, copy
# ${OUT} out with kubectl cp, then create ${OUT}/RELEASE so the pod exits (it exits on its own
# after 90 min). No prompt text is ever written to ${OUT}; the dataset lives only in /data.
set -uo pipefail

MODEL="${MODEL:-google/gemma-4-E4B-it}"
PORT=8000
BASE_URL="http://localhost:${PORT}"
OUT=/workspace/output
DATA=/data
SCRIPTS=/scripts
LG="${OUT}/loadgen"
mkdir -p "${OUT}" "${LG}"
export PYTHONPATH="${SCRIPTS}:${PYTHONPATH:-}"   # so --middleware=gdm_binary_probability.* imports

# The customer's command line, verbatim, with only the model name substituted.
# OMP_NUM_THREADS=8 and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True come from the manifest.
CUSTOMER_ARGS=(
  --tensor-parallel-size=1
  --load-format=runai_streamer
  --attention-backend=TRITON_ATTN
  --seed=42
  --gpu-memory-utilization=0.90
  --max-model-len=8192
  --max-num-seqs=256
  --max-num-batched-tokens=16384
  --kv-sharing-fast-prefill
  --max-logprobs=20
  --dtype=bfloat16
  --enable-prefix-caching
  --middleware=gdm_binary_probability.GdmBinaryProbabilityMiddleware
)

# Step lists. Overridable from the manifest (space-separated) so a relaunch after a spot
# preemption can skip what already completed, e.g. CONC_SWEEP="" RATE_STEPS="160 190 ...".
read -r -a CONC_SWEEP <<< "${CONC_SWEEP-1 4 8 12 16 20 24 32 48 64 96 128 256}"
read -r -a RATE_STEPS <<< "${RATE_STEPS-120 160 190 215 240 280 320}"
RATE_EXTRA_STEP=80
RATE_EXTRA_MAX=6
STEP_S=60
RATE_S=120
SOAK_S="${SOAK_S:-600}"
VARIANT_S=120
# If the rate steps were already measured in an earlier pod, the soak rate can be given directly.
SOAK_RATE_OVERRIDE="${SOAK_RATE_OVERRIDE:-}"

log () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${OUT}/progress.log"; }

{
  echo "vllm: $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1)"
  pip list 2>/dev/null | grep -i -E "^(vllm|torch|flashinfer|transformers|aiohttp|tokenizers|triton) "
} > "${OUT}/versions.txt"
nvidia-smi -q > "${OUT}/nvidia-smi-q.txt" 2>&1
env | grep -v -i -E "token|key|secret" | sort > "${OUT}/env.txt"
printf '%s\n' "vllm serve ${MODEL}" "${CUSTOMER_ARGS[@]}" > "${OUT}/server_args.txt"

nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used \
  --format=csv,nounits -l 1 > "${OUT}/gpu.csv" 2>&1 &
python3 "${SCRIPTS}/cpumon.py" --interval 1 --out "${OUT}/cpu.csv" &

SERVER_PID=""

stop_server () {
  if [ -n "${SERVER_PID}" ]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null
    for _ in $(seq 1 90); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null
    SERVER_PID=""
  fi
  pkill -KILL -f "[v]llm serve" 2>/dev/null
  pkill -KILL -f "[E]ngineCore" 2>/dev/null
  local used=""
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ "${used:-0}" -lt 2048 ] && return 0
    sleep 2
  done
  log "WARNING: GPU memory still at ${used} MiB after server shutdown"
}

# start_server <logname> <args...>; returns 1 if the server never became healthy.
start_server () {
  local name="$1"; shift
  log "server[${name}]: vllm serve ${MODEL} $*"
  setsid vllm serve "${MODEL}" "$@" > "${OUT}/server_${name}.log" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sf "${BASE_URL}/health" > /dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "server[${name}]: exited during startup; last lines:"
      tail -n 40 "${OUT}/server_${name}.log" | tee -a "${OUT}/progress.log"
      SERVER_PID=""
      stop_server
      return 1
    fi
    if [ "${waited}" -ge 2400 ]; then
      log "server[${name}]: not healthy after ${waited} s"
      stop_server
      return 1
    fi
    sleep 5; waited=$((waited + 5))
    if [ $((waited % 120)) -eq 0 ]; then log "server[${name}]: still starting (${waited} s)"; fi
  done
  log "server[${name}]: healthy after ${waited} s"
  grep -c -E "self_decoder|cross_decoder" "${OUT}/server_${name}.log" | xargs -I{} echo "fast-prefill compile markers: {}" | tee -a "${OUT}/progress.log"
}

# Last step's fields from steps.jsonl: last_field <key> [<subkey>]
last_field () {
  python3 - "$LG/steps.jsonl" "$@" <<'EOF'
import json, sys
path, keys = sys.argv[1], sys.argv[2:]
with open(path) as f:
    lines = [l for l in f if l.strip()]
v = json.loads(lines[-1]) if lines else {}
for k in keys:
    v = (v or {}).get(k) if isinstance(v, dict) else None
print("" if v is None else v)
EOF
}

# lg <label> <loadgen args...>
lg () {
  local label="$1"; shift
  python3 "${SCRIPTS}/rbx_loadgen.py" --base-url "${BASE_URL}" --model "${MODEL}" \
    --prompts "${PROMPTS}" --labels "${DATA}/labels.json" --outdir "${LG}" --label "${label}" "$@" \
    2>> "${LG}/stderr.log" | tee -a "${OUT}/progress.log"
  local rc=${PIPESTATUS[0]}
  [ "${rc}" -ne 0 ] && log "WARNING: loadgen ${label} exited ${rc}, see loadgen/stderr.log"
  sleep 3   # let the server drain before the next step's /metrics snapshot
  return 0
}

# ---------------------------------------------------------------- 0. server
if ! start_server customer "${CUSTOMER_ARGS[@]}"; then
  log "retrying without --load-format=runai_streamer (load path only; does not affect serving speed)"
  ARGS2=()
  for a in "${CUSTOMER_ARGS[@]}"; do [ "${a}" = "--load-format=runai_streamer" ] || ARGS2+=("${a}"); done
  if ! start_server customer_noruna "${ARGS2[@]}"; then
    log "FATAL: server would not start"; touch "${OUT}/DONE"; sleep 5400; exit 1
  fi
fi
curl -s "${BASE_URL}/v1/models" > "${OUT}/models.json"

# ---------------------------------------------------------------- 1. dataset
log "waiting for ${DATA}/READY (stream_dataset.py | kubectl exec ... ; then touch /data/READY)"
waited=0
until [ -f "${DATA}/READY" ]; do
  sleep 10; waited=$((waited + 10))
  if [ $((waited % 300)) -eq 0 ]; then log "still waiting for dataset (${waited} s)"; fi
  if [ "${waited}" -ge 5400 ]; then log "FATAL: no dataset after 90 min"; touch "${OUT}/DONE"; sleep 3600; exit 1; fi
done
log "dataset present: $(wc -l < "${DATA}/safechat.jsonl") rows; rendering prompts"
python3 "${SCRIPTS}/prep_dataset.py" --input "${DATA}/safechat.jsonl" --data-dir "${DATA}" \
  --stats "${OUT}/dataset_stats.json" > "${OUT}/prep.log" 2>&1 || log "WARNING: prep_dataset failed, see prep.log"
grep -E '"rows"|"duplicate|consecutive|"mean"|prefix_tokens' "${OUT}/prep.log" | head -20 | tee -a "${OUT}/progress.log"
PROMPTS="${DATA}/prompts_bracket.json"

# ---------------------------------------------------------------- 2. smoke
log "smoke test"
lg smoke --mode conc --concurrency 4 --max-requests 300 --duration 300
if [ "$(last_field n_ok)" = "0" ] || [ -z "$(last_field n_ok)" ]; then
  log "smoke returned no successful requests; retrying with middleware bypassed for diagnosis"
  lg smoke_bypass --mode conc --concurrency 4 --max-requests 100 --duration 120 --bypass-mw
  log "FATAL: smoke failed; stopping so logs can be inspected"
  touch "${OUT}/DONE"; sleep 5400; exit 1
fi
log "smoke: top1 is Safe/Un $(last_field top1_is_label_rate), both in top-20 $(last_field both_in_topk_rate), prompt tok mean $(last_field prompt_tokens mean), hit rate $(last_field metrics_delta hit_rate), auc $(last_field p_unsafe auc_vs_label)"

# Throwaway load so compile/autotune on new batch shapes never lands in a measured step.
lg warmup_c64 --mode conc --concurrency 64 --duration 30 --warmup 0

# ---------------------------------------------------------------- 3. closed-loop sweep
log "closed-loop sweep: ${CONC_SWEEP[*]-} x ${STEP_S} s"
for c in ${CONC_SWEEP[@]+"${CONC_SWEEP[@]}"}; do
  lg "conc_c${c}" --mode conc --concurrency "${c}" --duration "${STEP_S}"
done

# ---------------------------------------------------------------- 4. open-loop rate steps
log "open-loop Poisson steps: ${RATE_STEPS[*]-} req/s x ${RATE_S} s"
LAST_RATE="${LAST_RATE_DONE:-0}"   # highest rate already measured in an earlier pod, if resuming
for r in ${RATE_STEPS[@]+"${RATE_STEPS[@]}"}; do
  lg "rate_${r}" --mode rate --rate "${r}" --duration "${RATE_S}"
  LAST_RATE=${r}
done
extra=0
while [ "${extra}" -lt "${RATE_EXTRA_MAX}" ] && [ "${#RATE_STEPS[@]}" -gt 0 ]; do
  p50=$(last_field lat_ms p50); ach=$(last_field req_per_s); nerr=$(last_field n_err)
  stop=$(python3 -c "import sys; p,a,o,e=float(sys.argv[1] or 1e9),float(sys.argv[2] or 0),float(sys.argv[3]),int(sys.argv[4] or 0); print(int(p>150 or a<0.9*o or e>50))" "${p50}" "${ach}" "${LAST_RATE}" "${nerr}")
  [ "${stop}" = "1" ] && break
  LAST_RATE=$((LAST_RATE + RATE_EXTRA_STEP)); extra=$((extra + 1))
  lg "rate_${LAST_RATE}" --mode rate --rate "${LAST_RATE}" --duration "${RATE_S}"
done

# ---------------------------------------------------------------- 5. soak at p50 ~ 70 ms
if [ -n "${SOAK_RATE_OVERRIDE}" ]; then
  SOAK_RATE="${SOAK_RATE_OVERRIDE}"
else
  SOAK_RATE=$(python3 "${SCRIPTS}/pick_rate.py" "${LG}/steps.jsonl" --prefix rate_ --target-p50-ms 70 2>>"${LG}/stderr.log" || echo 200)
fi
log "soak: ${SOAK_S} s at ${SOAK_RATE} req/s (rate where p50 ~ 70 ms)"
lg "soak_${SOAK_RATE}" --mode rate --rate "${SOAK_RATE}" --duration "${SOAK_S}"

# ---------------------------------------------------------------- 6. variants at the soak rate
log "variants at ${SOAK_RATE} req/s x ${VARIANT_S} s"
lg "var_fileorder_${SOAK_RATE}"   --mode rate --rate "${SOAK_RATE}" --duration "${VARIANT_S}" --order file
PROMPTS="${DATA}/prompts_plain.json"
lg "var_plainfmt_${SOAK_RATE}"    --mode rate --rate "${SOAK_RATE}" --duration "${VARIANT_S}"
PROMPTS="${DATA}/prompts_bracket.json"
lg "var_completions_${SOAK_RATE}" --mode rate --rate "${SOAK_RATE}" --duration "${VARIANT_S}" --endpoint completions
lg "var_bypassmw_${SOAK_RATE}"    --mode rate --rate "${SOAK_RATE}" --duration "${VARIANT_S}" --bypass-mw
# The stand-in middleware injects allowed_token_ids itself, so "no allowed ids" needs the bypass;
# compare it against var_bypassmw, not against the base.
lg "var_noallowed_${SOAK_RATE}"   --mode rate --rate "${SOAK_RATE}" --duration "${VARIANT_S}" --bypass-mw --no-allowed
# Same variants in closed loop at c=16 (near the customer's latency on our earlier curve).
lg "var_c16_base"       --mode conc --concurrency 16 --duration "${STEP_S}"
lg "var_c16_bypassmw"   --mode conc --concurrency 16 --duration "${STEP_S}" --bypass-mw
lg "var_c16_noallowed"  --mode conc --concurrency 16 --duration "${STEP_S}" --bypass-mw --no-allowed

# ---------------------------------------------------------------- 7. tie-back to the synthetic spec
log "tie-back: vllm bench serve, random 50+450 tokens, 1 output token (our earlier B config gave ~136 req/s at saturation)"
mkdir -p "${OUT}/tieback"
for c in 8 16 64 128; do
  n=$(( c * 20 > 1000 ? c * 20 : 1000 ))
  curl -s "${BASE_URL}/metrics" > "${OUT}/tieback/c${c}_before.prom"
  CUDA_VISIBLE_DEVICES="" vllm bench serve --model "${MODEL}" --dataset-name random \
    --random-input-len 450 --random-prefix-len 50 --random-output-len 1 --random-range-ratio 0.0 \
    --num-prompts "${n}" --seed $((2000 + c)) --max-concurrency "${c}" \
    --header x-gdm-bypass=1 --metric-percentiles 50,90,99 --percentile-metrics ttft,e2el \
    --save-result --result-dir "${OUT}/tieback" --result-filename "c${c}.json" \
    --base-url "${BASE_URL}" > "${OUT}/tieback/c${c}_bench.log" 2>&1 \
    || log "WARNING: tie-back c=${c} failed, see tieback/c${c}_bench.log"
  curl -s "${BASE_URL}/metrics" > "${OUT}/tieback/c${c}_after.prom"
  python3 "${SCRIPTS}/metrics_delta.py" "${OUT}/tieback/c${c}_before.prom" "${OUT}/tieback/c${c}_after.prom" \
    --label "tieback/c${c}" --num-prompts "${n}" >> "${OUT}/tieback/summary.jsonl"
  rps=$(grep -m1 "Request throughput" "${OUT}/tieback/c${c}_bench.log" | awk '{print $NF}')
  p50=$(grep -m1 "Median E2EL" "${OUT}/tieback/c${c}_bench.log" | awk '{print $NF}')
  p99=$(grep -m1 "P99 E2EL" "${OUT}/tieback/c${c}_bench.log" | awk '{print $NF}')
  log "    tie-back c=${c}: ${rps} req/s, p50 ${p50} ms, p99 ${p99} ms"
done

log "ALL DONE"
touch "${OUT}/DONE"
for _ in $(seq 1 1080); do
  [ -f "${OUT}/RELEASE" ] && break
  sleep 5
done
stop_server
log "released, exiting"
