#!/usr/bin/env bash
# Copy the pod's output dir out every 10 min (spot insurance) until DONE is seen or the pod is gone.
# Usage: copy_loop.sh <results dir>      (NS / JOB / INTERVAL env overrides)
set -uo pipefail
NS="${NS:-rbx-bench}"
JOB="${JOB:-rbx-replay-v6e}"
INTERVAL="${INTERVAL:-300}"
DEST="${1:?results dir}"
TMP="/tmp/rbx-v6e-copy-$$"
mkdir -p "${DEST}"
while :; do
  POD=$(kubectl -n "${NS}" get pods -l job-name="${JOB}" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  PHASE=$(kubectl -n "${NS}" get pod "${POD}" -o jsonpath='{.status.phase}' 2>/dev/null)
  if [ -z "${POD}" ] || [ "${PHASE}" = "Failed" ] || [ "${PHASE}" = "Succeeded" ]; then
    echo "[$(date -u +%H:%M:%S)] pod ${POD:-none} phase ${PHASE:-none}; stopping"; exit 2
  fi
  rm -rf "${TMP}"
  if kubectl -n "${NS}" cp "${POD}:/workspace/output" "${TMP}" > /dev/null 2>&1; then
    rsync -a "${TMP}/" "${DEST}/"
    echo "[$(date -u +%H:%M:%S)] copied: $(du -sh "${DEST}" | cut -f1); last: $(tail -n 1 "${DEST}/progress.log" 2>/dev/null)"
    if [ -f "${DEST}/DONE" ]; then echo "DONE seen; final copy complete"; rm -rf "${TMP}"; exit 0; fi
  else
    echo "[$(date -u +%H:%M:%S)] kubectl cp failed (pod starting?)"
  fi
  sleep "${INTERVAL}"
done
