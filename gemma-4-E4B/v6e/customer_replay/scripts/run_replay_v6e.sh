#!/usr/bin/env bash
# Customer replay: safechat-100k against vLLM-TPU with the customer's server flags, on ONE TPU v6e
# chip (ct6e-standard-1t). The 1:1 counterpart of ../../g4/customer_replay/scripts/run_replay.sh
# (one RTX PRO 6000 on g4-standard-48). Same data, same client (rbx_loadgen.py, unchanged), same
# step durations and readouts; only the accelerator and the CUDA-only flags differ.
#
# Differences from the G4 driver, all deliberate:
#   * --load-format=runai_streamer and --attention-backend=TRITON_ATTN dropped (CUDA-only / load
#     path only); --data-parallel-size=1 made explicit.
#   * --block-size=32 added: tpu-inference's default page size at --max-model-len 8192 is 256
#     tokens, which would cache nothing of the 98-token shared prefix (27% hit rate on G4 with
#     32-token blocks). Falls back to 64, 128, then the TPU default if 32 does not start.
#   * --kv-sharing-fast-prefill is kept but is expected to be a no-op (tpu-inference has no
#     fast-prefill implementation); variant V1_nofp measures it off.
#   * Variant V1_bs256 measures the TPU default block size (prefix cache effectively off).
#   * Poisson rates are derived from the measured closed-loop plateau (50..95%) instead of the
#     G4-specific absolute list; the readout (rate at p50 = 70 ms) is interpolated as before.
#   * tpu-info snapshots replace nvidia-smi; TPU-aware server shutdown (chip holders, lockfile).
#   * allowed_token_ids is NOT sent (ALLOWED_TOKEN_IDS=0 default): tpu-inference's sampler path for it
#     crashes the engine (input_batch.py:451 TypeError, present on main as of 2026-10-09). The score
#     needs only logP(Un) and logP(Safe) from the top-20, which the mask does not change; on G4 the
#     flag cost nothing measurable (c=16: 194.9 vs 198.3 req/s). The client passes --no-allowed and
#     the stand-in middleware strips it (GDM_NO_ALLOWED_TOKEN_IDS=1).
#
# Sequence:
#   V1        server (customer flags) -> dataset -> smoke -> warmup -> closed loop c=1..256 x 60 s
#             -> Poisson steps x 120 s -> 10-min soak at p50~70 ms -> variants at the soak rate and
#             at c=16 -> synthetic tie-back (vllm bench serve, 50+450 tokens) -> stop
#   V1_nofp   server without --kv-sharing-fast-prefill: c=8..128 x 60 s + Poisson at the soak rate
#   V1_bs256  server without --block-size: c=8..128 x 60 s
#
# Results stay in ${OUT} (no cloud credentials in the pod). When ${OUT}/DONE appears, copy ${OUT}
# out with kubectl cp, then create ${OUT}/RELEASE so the pod exits (it exits on its own after
# 90 min). No prompt text is ever written to ${OUT}; the dataset lives only in /data (tmpfs).
#
# Env overrides (resume after a spot preemption, or trim): VARIANTS="V1 V1_nofp V1_bs256",
# CONC_SWEEP, RATE_STEPS (absolute, skips derivation), RATE_FRACS, SOAK_S, SOAK_RATE_OVERRIDE,
# LAST_RATE_DONE, VAR_CONC, TIEBACK=0, BLOCK_SIZE_CANDIDATES, TEXT_ONLY=1, START_TIMEOUT_S, MODEL.
set -uo pipefail

MODEL="${MODEL:-google/gemma-4-E4B-it}"
PORT=8000
BASE_URL="http://localhost:${PORT}"
OUT="${OUT:-/workspace/output}"
DATA="${DATA:-/data}"
SCRIPTS="${SCRIPTS:-/scripts}"
LG="${OUT}/loadgen"
mkdir -p "${OUT}" "${LG}"
export PYTHONPATH="${SCRIPTS}:${PYTHONPATH:-}"   # so --middleware=gdm_binary_probability.* imports

# The customer's command line with only the CUDA-only flags removed (see header). Order kept.
# --block-size and --kv-sharing-fast-prefill are appended per variant by server_args().
BASE_ARGS=(
  --tensor-parallel-size=1
  --data-parallel-size=1
  --seed=42
  --gpu-memory-utilization=0.90
  --max-model-len=8192
  --max-num-seqs=256
  --max-num-batched-tokens=16384
  --max-logprobs=20
  --dtype=bfloat16
  --enable-prefix-caching
  --middleware=gdm_binary_probability.GdmBinaryProbabilityMiddleware
)
# Only if the first start fails or hangs: text-only mode skips the vision/audio encoders
# (startup time only; text serving speed is unchanged). Set TEXT_ONLY=1 on relaunch.
TEXT_ONLY_ARGS=(--limit-mm-per-prompt '{"image":0,"audio":0}')

VARIANTS="${VARIANTS:-V1 V1_nofp V1_bs256}"
read -r -a CONC_SWEEP <<< "${CONC_SWEEP-1 4 8 12 16 20 24 32 48 64 96 128 256}"
read -r -a RATE_STEPS <<< "${RATE_STEPS-}"
read -r -a RATE_FRACS <<< "${RATE_FRACS-0.5 0.6 0.7 0.8 0.9 0.95}"
read -r -a RATE_DOWN_FRACS <<< "${RATE_DOWN_FRACS-0.4 0.3 0.2}"   # used only if p50 > 70 ms at the first rate
read -r -a VAR_CONC <<< "${VAR_CONC-8 16 32 64 128}"
read -r -a BLOCK_SIZE_CANDIDATES <<< "${BLOCK_SIZE_CANDIDATES-32 64 128 default}"
STEP_S="${STEP_S:-60}"
RATE_S="${RATE_S:-120}"
SOAK_S="${SOAK_S:-600}"
VARIANT_S="${VARIANT_S:-120}"
WARM_S="${WARM_S:-30}"
TIEBACK="${TIEBACK:-1}"
SOAK_RATE_OVERRIDE="${SOAK_RATE_OVERRIDE:-}"
START_TIMEOUT_S="${START_TIMEOUT_S:-2400}"
SOAK_RATE=""
ALLOWED_TOKEN_IDS="${ALLOWED_TOKEN_IDS:-0}"
LG_EXTRA=()
if [ "${ALLOWED_TOKEN_IDS}" = "0" ]; then
  LG_EXTRA=(--no-allowed)
  export GDM_NO_ALLOWED_TOKEN_IDS=1
fi

log () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "${OUT}/progress.log"; }

{
  echo "image: ${IMAGE_REF:-unknown}"
  echo "node: ${NODE_NAME:-unknown}"
  echo "date: $(date -u +%FT%TZ)"
  JAX_PLATFORMS=cpu python3 - <<'EOF'
import importlib.metadata as m
for name in ("vllm", "tpu_inference", "tpu-inference", "jax", "jaxlib", "libtpu", "libtpu-nightly", "flax",
             "torch", "torchax", "transformers", "tokenizers", "aiohttp", "tpu-info"):
    try:
        print(f"{name} {m.version(name)}")
    except Exception:
        pass
EOF
  echo "nproc: $(nproc)"
  free -g | head -n 2
  ls -l /dev/vfio /dev/accel* 2>&1
} > "${OUT}/versions.txt" 2>&1
env | grep -v -i -E "token|key|secret" | sort > "${OUT}/env.txt"

# ---------------------------------------------------------------- telemetry
python3 "${SCRIPTS}/cpumon.py" --interval 1 --out "${OUT}/cpu.csv" > "${OUT}/cpumon.log" 2>&1 &

# tpu-info every 5 s (chip table: HBM usage, duty cycle). Best effort; never fails anything.
if command -v tpu-info > /dev/null 2>&1; then
  ( while :; do
      { echo "=== $(date -u +%FT%T.%3NZ) unix=$(date +%s.%N)"; timeout 20 tpu-info 2>&1; } >> "${OUT}/tpu_info.log"
      sleep 5
    done ) &
else
  log "tpu-info not found in the image; no chip telemetry (HBM/duty cycle) will be recorded"
fi

# ---------------------------------------------------------------- server control
SERVER_PID=""

tpu_holders () {
  local p
  for p in /proc/[0-9]*; do
    ls -l "${p}/fd" 2>/dev/null | grep -q -E -- '-> /dev/(vfio/[0-9]+|accel[0-9]+)$' && echo "${p#/proc/}"
  done
}

stop_server () {
  if [ -n "${SERVER_PID}" ]; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null
    for _ in $(seq 1 120); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null
    SERVER_PID=""
  fi
  pkill -KILL -f "[v]llm serve" 2>/dev/null
  pkill -KILL -f "[V]LLM::" 2>/dev/null
  pkill -KILL -f "[E]ngineCore" 2>/dev/null
  local holders="" i
  for i in $(seq 1 90); do
    holders=$(tpu_holders | tr '\n' ' ')
    if [ -z "${holders// /}" ] && ! curl -s -o /dev/null --max-time 2 "${BASE_URL}/health"; then
      rm -f /tmp/libtpu_lockfile
      sleep 10
      log "server stopped: no TPU holders, port ${PORT} closed (checked ${i}x)"
      return 0
    fi
    # shellcheck disable=SC2086
    [ -n "${holders// /}" ] && kill -KILL ${holders} 2>/dev/null
    sleep 2
  done
  log "WARNING: TPU still held by [${holders}] after shutdown"
  return 1
}

# Effective config lines from the server log (vLLM's config dump + tpu-inference's block-size line).
effective_config () {
  local f="$1"
  {
    grep -o -E "block_size=[0-9]+|kv_sharing_fast_prefill=(True|False)|enable_prefix_caching=(True|False)|max_num_batched_tokens=[0-9]+|max_num_seqs=[0-9]+" "${f}" | sort -u | tr '\n' ' '
    grep -o -E "Using cache_config.block_size: [0-9]+|Resolved TPU_MULTIPROCESS_DP=[a-zA-Z0-9]+|page size from [0-9]+ to [0-9]+" "${f}" | sort -u | tr '\n' ';'
  } 2>/dev/null
}

# start_server <logname> <args...>; returns 1 if the server never became healthy.
start_server () {
  local name="$1"; shift
  log "server[${name}]: vllm serve ${MODEL} $*"
  printf '%s\n' "vllm serve ${MODEL}" "$@" > "${OUT}/server_args_${name}.txt"
  local t0
  t0=$(date +%s)
  setsid vllm serve "${MODEL}" "$@" > "${OUT}/server_${name}.log" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sf --max-time 5 "${BASE_URL}/health" > /dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "server[${name}]: exited during startup after $(( $(date +%s) - t0 )) s; last lines:"
      tail -n 40 "${OUT}/server_${name}.log" | tee -a "${OUT}/progress.log"
      SERVER_PID=""
      stop_server
      return 1
    fi
    if [ "${waited}" -ge "${START_TIMEOUT_S}" ]; then
      log "server[${name}]: not healthy after ${waited} s; last lines:"
      tail -n 20 "${OUT}/server_${name}.log" | tee -a "${OUT}/progress.log"
      stop_server
      return 1
    fi
    sleep 5; waited=$((waited + 5))
    if [ $((waited % 120)) -eq 0 ]; then log "server[${name}]: still starting (${waited} s)"; fi
  done
  log "server[${name}]: healthy after $(( $(date +%s) - t0 )) s"
  log "server[${name}]: effective config: $(effective_config "${OUT}/server_${name}.log")"
  curl -s --max-time 30 "${BASE_URL}/v1/models" > "${OUT}/models_${name}.json"
  curl -s --max-time 30 "${BASE_URL}/metrics" > "${OUT}/metrics_at_start_${name}.prom"
  grep -c -i -E "fast_prefill|fast-prefill" "${OUT}/server_${name}.log" | xargs -I{} echo "server[${name}]: fast-prefill mentions in log: {}" | tee -a "${OUT}/progress.log"
  return 0
}

# server_args <fastprefill 0|1> <block_size|default> -> prints one arg per line
server_args () {
  local fp="$1" bs="$2" a
  for a in "${BASE_ARGS[@]}"; do echo "${a}"; done
  [ "${fp}" = "1" ] && echo "--kv-sharing-fast-prefill"
  [ "${bs}" != "default" ] && echo "--block-size=${bs}"
  if [ "${TEXT_ONLY:-0}" = "1" ]; then for a in "${TEXT_ONLY_ARGS[@]}"; do echo "${a}"; done; fi
  return 0
}

# start_with_fallbacks <name> <fastprefill 0|1> <block_size candidates...>
# Tries each block size in order; sets EFFECTIVE_BS. Returns 1 if none started.
EFFECTIVE_BS=""
start_with_fallbacks () {
  local name="$1" fp="$2"; shift 2
  local bs args=()
  for bs in "$@"; do
    args=()
    while IFS= read -r a; do args+=("${a}"); done < <(server_args "${fp}" "${bs}")
    if start_server "${name}_bs${bs}" "${args[@]}"; then
      EFFECTIVE_BS="${bs}"
      [ "${bs}" != "$1" ] && log "server[${name}]: NOTE block size ${bs} used instead of $1"
      return 0
    fi
    log "server[${name}]: start failed with block size ${bs}"
  done
  return 1
}

# ---------------------------------------------------------------- helpers
last_field () {
  python3 - "$LG/steps.jsonl" "$@" <<'EOF'
import json, sys
path, keys = sys.argv[1], sys.argv[2:]
try:
    with open(path) as f:
        lines = [l for l in f if l.strip()]
except FileNotFoundError:
    lines = []
v = json.loads(lines[-1]) if lines else {}
for k in keys:
    v = (v or {}).get(k) if isinstance(v, dict) else None
print("" if v is None else v)
EOF
}

# Highest closed-loop throughput measured so far (labels conc_c*), rounded down to an integer.
plateau_rps () {
  python3 - "$LG/steps.jsonl" <<'EOF'
import json, sys
best = 0.0
try:
    for line in open(sys.argv[1]):
        if not line.strip():
            continue
        s = json.loads(line)
        if s.get("mode") == "conc" and s["label"].startswith("conc_c") and s.get("req_per_s"):
            best = max(best, float(s["req_per_s"]))
except FileNotFoundError:
    pass
print(int(best))
EOF
}

round5 () { python3 -c "import sys; print(max(5, int(5 * round(float(sys.argv[1]) / 5.0))))" "$1"; }

# lg <label> <loadgen args...>   (client processes never touch the TPU: JAX_PLATFORMS=cpu)
lg () {
  local label="$1"; shift
  JAX_PLATFORMS=cpu python3 "${SCRIPTS}/rbx_loadgen.py" --base-url "${BASE_URL}" --model "${MODEL}" \
    --prompts "${PROMPTS}" --labels "${DATA}/labels.json" --outdir "${LG}" --label "${label}" "$@" ${LG_EXTRA[@]+"${LG_EXTRA[@]}"} \
    2>> "${LG}/stderr.log" | tee -a "${OUT}/progress.log"
  local rc=${PIPESTATUS[0]}
  [ "${rc}" -ne 0 ] && log "WARNING: loadgen ${label} exited ${rc}, see loadgen/stderr.log"
  sleep 3   # let the server drain before the next step's /metrics snapshot
  return 0
}

tieback () {
  log "tie-back: vllm bench serve, random 50+450 tokens, 1 output token (G4 1-GPU B config gave ~136 req/s at saturation)"
  mkdir -p "${OUT}/tieback"
  local c n rps p50 p99
  for c in 8 16 64 128; do
    n=$(( c * 20 > 1000 ? c * 20 : 1000 ))
    curl -s "${BASE_URL}/metrics" > "${OUT}/tieback/c${c}_before.prom"
    JAX_PLATFORMS=cpu timeout 900 vllm bench serve --model "${MODEL}" --dataset-name random \
      --random-input-len 450 --random-prefix-len 50 --random-output-len 1 --random-range-ratio 0.0 \
      --num-prompts "${n}" --seed $((2000 + c)) --max-concurrency "${c}" \
      --header x-gdm-bypass=1 --metric-percentiles 50,90,99 --percentile-metrics ttft,e2el \
      --save-result --result-dir "${OUT}/tieback" --result-filename "c${c}.json" \
      --base-url "${BASE_URL}" > "${OUT}/tieback/c${c}_bench.log" 2>&1 \
      || log "WARNING: tie-back c=${c} failed, see tieback/c${c}_bench.log"
    curl -s "${BASE_URL}/metrics" > "${OUT}/tieback/c${c}_after.prom"
    python3 "${SCRIPTS}/metrics_delta.py" "${OUT}/tieback/c${c}_before.prom" "${OUT}/tieback/c${c}_after.prom" \
      --label "tieback/c${c}" --num-prompts "${n}" >> "${OUT}/tieback/summary.jsonl" 2>/dev/null
    rps=$(grep -m1 "Request throughput" "${OUT}/tieback/c${c}_bench.log" | awk '{print $NF}')
    p50=$(grep -m1 "Median E2EL" "${OUT}/tieback/c${c}_bench.log" | awk '{print $NF}')
    p99=$(grep -m1 "P99 E2EL" "${OUT}/tieback/c${c}_bench.log" | awk '{print $NF}')
    log "    tie-back c=${c}: ${rps} req/s, p50 ${p50} ms, p99 ${p99} ms"
  done
}

finish () {
  log "ALL DONE"
  touch "${OUT}/DONE"
  for _ in $(seq 1 1080); do
    [ -f "${OUT}/RELEASE" ] && break
    sleep 5
  done
  stop_server
  log "released, exiting"
  exit "${1:-0}"
}

trap 'log "SIGTERM received, stopping"; stop_server; exit 143' TERM

log "start: variants [${VARIANTS}], image ${IMAGE_REF:-unknown}, node ${NODE_NAME:-unknown}, chips: $(ls /dev/vfio 2>/dev/null | tr '\n' ' '), allowed_token_ids=${ALLOWED_TOKEN_IDS}"

# ================================================================ V1: customer config
run_V1 () {
  if ! start_with_fallbacks V1 1 "${BLOCK_SIZE_CANDIDATES[@]}"; then
    log "FATAL: V1 server would not start with any block size"
    finish 1
  fi

  # ---------------------------------------------------------------- dataset
  if [ ! -f "${DATA}/prompts_bracket.json" ]; then
    log "waiting for ${DATA}/READY (local/stream.sh streams the dataset, checks the row count, touches READY)"
    local waited=0
    until [ -f "${DATA}/READY" ]; do
      sleep 10; waited=$((waited + 10))
      if [ $((waited % 300)) -eq 0 ]; then log "still waiting for dataset (${waited} s)"; fi
      if [ "${waited}" -ge 5400 ]; then log "FATAL: no dataset after 90 min"; finish 1; fi
    done
    log "dataset present: $(wc -l < "${DATA}/safechat.jsonl") rows; rendering prompts"
    python3 "${SCRIPTS}/prep_dataset.py" --input "${DATA}/safechat.jsonl" --data-dir "${DATA}" \
      --stats "${OUT}/dataset_stats.json" > "${OUT}/prep.log" 2>&1 || log "WARNING: prep_dataset failed, see prep.log"
    grep -E '"rows"|"duplicate|consecutive|"mean"|prefix_tokens' "${OUT}/prep.log" | head -20 | tee -a "${OUT}/progress.log"
  fi
  PROMPTS="${DATA}/prompts_bracket.json"

  # ---------------------------------------------------------------- smoke
  log "smoke test"
  lg smoke --mode conc --concurrency 4 --max-requests 300 --duration 300
  if [ "$(last_field n_ok)" = "0" ] || [ -z "$(last_field n_ok)" ]; then
    log "smoke returned no successful requests; retrying with middleware bypassed, then without allowed_token_ids"
    lg smoke_bypass --mode conc --concurrency 4 --max-requests 100 --duration 120 --bypass-mw
    [ "${ALLOWED_TOKEN_IDS}" = "1" ] && lg smoke_noallowed --mode conc --concurrency 4 --max-requests 100 --duration 120 --bypass-mw --no-allowed
    log "FATAL: smoke failed; stopping so logs can be inspected (loadgen/stderr.log, server_V1*.log)"
    finish 1
  fi
  log "smoke: top1 is Safe/Un $(last_field top1_is_label_rate), both in top-20 $(last_field both_in_topk_rate), prompt tok mean $(last_field prompt_tokens mean), hit rate $(last_field metrics_delta hit_rate), auc $(last_field p_unsafe auc_vs_label)"

  # Throwaway load so compile/autotune on new batch shapes never lands in a measured step.
  lg warmup_c64 --mode conc --concurrency 64 --duration "${WARM_S}" --warmup 0
  lg warmup_c256 --mode conc --concurrency 256 --duration "${WARM_S}" --warmup 0

  # ---------------------------------------------------------------- closed-loop sweep
  log "closed-loop sweep: ${CONC_SWEEP[*]-} x ${STEP_S} s"
  local c
  for c in ${CONC_SWEEP[@]+"${CONC_SWEEP[@]}"}; do
    lg "conc_c${c}" --mode conc --concurrency "${c}" --duration "${STEP_S}"
  done

  # ---------------------------------------------------------------- open-loop Poisson steps
  local plateau r p50 ach
  plateau=$(plateau_rps)
  if [ "${#RATE_STEPS[@]}" -eq 0 ] && [ "${plateau}" -gt 0 ]; then
    for f in "${RATE_FRACS[@]}"; do RATE_STEPS+=("$(round5 "$(python3 -c "print(${f} * ${plateau})")")"); done
  fi
  log "open-loop Poisson steps: ${RATE_STEPS[*]-} req/s x ${RATE_S} s (closed-loop plateau ${plateau} req/s)"
  local first=1
  for r in ${RATE_STEPS[@]+"${RATE_STEPS[@]}"}; do
    lg "rate_${r}" --mode rate --rate "${r}" --duration "${RATE_S}"
    if [ "${first}" = 1 ]; then
      first=0
      p50=$(last_field lat_ms p50)
      # Already above the target at the lowest rate: probe lower rates so pick_rate can interpolate.
      if python3 -c "import sys; sys.exit(0 if float(sys.argv[1] or 0) > 70 else 1)" "${p50}"; then
        for f in "${RATE_DOWN_FRACS[@]}"; do
          r=$(round5 "$(python3 -c "print(${f} * ${plateau})")")
          lg "rate_${r}" --mode rate --rate "${r}" --duration "${RATE_S}"
          p50=$(last_field lat_ms p50)
          python3 -c "import sys; sys.exit(0 if float(sys.argv[1] or 0) <= 70 else 1)" "${p50}" && break
        done
      fi
    fi
  done

  # ---------------------------------------------------------------- soak at p50 ~ 70 ms
  if [ -n "${SOAK_RATE_OVERRIDE}" ]; then
    SOAK_RATE="${SOAK_RATE_OVERRIDE}"
  else
    SOAK_RATE=$(python3 "${SCRIPTS}/pick_rate.py" "${LG}/steps.jsonl" --prefix rate_ --target-p50-ms 70 2>>"${LG}/stderr.log" || echo "${plateau:-100}")
  fi
  log "soak: ${SOAK_S} s at ${SOAK_RATE} req/s (rate where p50 ~ 70 ms)"
  lg "soak_${SOAK_RATE}" --mode rate --rate "${SOAK_RATE}" --duration "${SOAK_S}"

  # ---------------------------------------------------------------- variants at the soak rate / c=16
  log "variants at ${SOAK_RATE} req/s x ${VARIANT_S} s and at c=16"
  PROMPTS="${DATA}/prompts_plain.json"
  lg "var_plainfmt_${SOAK_RATE}" --mode rate --rate "${SOAK_RATE}" --duration "${VARIANT_S}"
  PROMPTS="${DATA}/prompts_bracket.json"
  lg "var_bypassmw_${SOAK_RATE}" --mode rate --rate "${SOAK_RATE}" --duration "${VARIANT_S}" --bypass-mw
  lg "var_c16_base"      --mode conc --concurrency 16 --duration "${STEP_S}"
  lg "var_c16_bypassmw"  --mode conc --concurrency 16 --duration "${STEP_S}" --bypass-mw
  # The stand-in middleware injects allowed_token_ids itself, so "no allowed ids" needs the bypass;
  # compare it against var_c16_bypassmw, not against the base.
  [ "${ALLOWED_TOKEN_IDS}" = "1" ] && lg "var_c16_noallowed" --mode conc --concurrency 16 --duration "${STEP_S}" --bypass-mw --no-allowed

  # ---------------------------------------------------------------- tie-back to the synthetic spec
  [ "${TIEBACK}" = "1" ] && tieback

  curl -s --max-time 30 "${BASE_URL}/metrics" > "${OUT}/metrics_at_end_V1.prom"
  stop_server
  touch "${OUT}/V1_COMPLETE"
  log "V1 complete"
}

# ================================================================ short variants (own server each)
# run_short <name> <fastprefill 0|1> <block_size|default> <with_poisson 0|1>
run_short () {
  local name="$1" fp="$2" bs="$3" withrate="$4" c rate
  if ! start_with_fallbacks "${name}" "${fp}" "${bs}"; then
    log "WARNING: ${name} server would not start; skipping variant"
    touch "${OUT}/${name}_COMPLETE"
    return 1
  fi
  PROMPTS="${DATA}/prompts_bracket.json"
  lg "${name}_warmup_c64" --mode conc --concurrency 64 --duration "${WARM_S}" --warmup 0
  log "${name}: closed loop ${VAR_CONC[*]-} x ${STEP_S} s"
  for c in ${VAR_CONC[@]+"${VAR_CONC[@]}"}; do
    lg "${name}_conc_c${c}" --mode conc --concurrency "${c}" --duration "${STEP_S}"
  done
  if [ "${withrate}" = "1" ]; then
    rate="${SOAK_RATE:-${SOAK_RATE_OVERRIDE}}"
    if [ -n "${rate}" ]; then
      lg "${name}_rate_${rate}" --mode rate --rate "${rate}" --duration "${RATE_S}"
    else
      log "${name}: no soak rate known (V1 not run in this pod and SOAK_RATE_OVERRIDE unset); skipping the Poisson step"
    fi
  fi
  curl -s --max-time 30 "${BASE_URL}/metrics" > "${OUT}/metrics_at_end_${name}.prom"
  stop_server
  touch "${OUT}/${name}_COMPLETE"
  log "${name} complete"
}

for v in ${VARIANTS}; do
  case "${v}" in
    V1)       run_V1 ;;
    V1_nofp)  run_short V1_nofp 0 "${EFFECTIVE_BS:-${BLOCK_SIZE_CANDIDATES[0]}}" 1 ;;
    V1_bs256) run_short V1_bs256 1 default 0 ;;
    *)        log "WARNING: unknown variant ${v}" ;;
  esac
done

finish 0
