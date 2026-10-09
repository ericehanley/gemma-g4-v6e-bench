#!/usr/bin/env bash
# Wait until the pod's progress.log matches a regex (milestone) or a new failure line appears.
# Usage: watch.sh '<milestone regex>' [timeout seconds]     (NS / JOB env overrides)
# Exit 0 = milestone, 1 = timeout, 2 = pod gone/failed, 3 = new failure line, 4 = still Pending.
PATTERN="$1"; TIMEOUT="${2:-3600}"
FAIL='FATAL|exited during startup|not healthy after|WARNING: .* would not start|start failed with block size'
NS="${NS:-rbx-bench}"
JOB="${JOB:-rbx-replay-v6e}"
BASE=""; LOG=""; waited=0
while [ "${waited}" -lt "${TIMEOUT}" ]; do
  POD=$(kubectl -n "${NS}" get pods -l job-name="${JOB}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  PHASE=$(kubectl -n "${NS}" get pod "${POD}" -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ -z "${POD}" ] || [ "${PHASE}" = "Failed" ] || [ "${PHASE}" = "Succeeded" ]; then
    echo "POD=${POD:-none} PHASE=${PHASE:-none}"; kubectl -n "${NS}" get pods -l job-name="${JOB}" -o wide
    kubectl -n "${NS}" describe pod "${POD}" 2>/dev/null | tail -n 20
    exit 2
  fi
  if [ "${PHASE}" = "Pending" ]; then
    if [ $((waited % 300)) -eq 0 ]; then
      echo "[${waited}s] Pending: $(kubectl -n "${NS}" get events --field-selector "involvedObject.name=${POD}" --sort-by=.lastTimestamp 2>/dev/null | tail -n 2 | awk '{$1=$2=$3="";print}')"
    fi
  elif LOG=$(kubectl -n "${NS}" exec "${POD}" -- cat /workspace/output/progress.log 2>/dev/null); then
    [ -z "${BASE}" ] && BASE=$(printf '%s\n' "${LOG}" | wc -l)
    if printf '%s\n' "${LOG}" | tail -n +"$((BASE + 1))" | grep -q -E "${FAIL}"; then
      echo "NEW FAILURE LINE (pod ${POD}, waited ${waited}s):"; printf '%s\n' "${LOG}" | tail -n 30; exit 3
    fi
    if printf '%s\n' "${LOG}" | grep -q -E "${PATTERN}"; then
      echo "MILESTONE '${PATTERN}' (pod ${POD}, node $(kubectl -n "${NS}" get pod "${POD}" -o jsonpath='{.spec.nodeName}'), waited ${waited}s):"
      printf '%s\n' "${LOG}" | tail -n 30; exit 0
    fi
  fi
  sleep 30; waited=$((waited + 30))
done
echo "TIMEOUT after ${TIMEOUT}s waiting for '${PATTERN}' (phase ${PHASE:-none})"; printf '%s\n' "${LOG}" | tail -n 30
[ "${PHASE}" = "Pending" ] && exit 4
exit 1
