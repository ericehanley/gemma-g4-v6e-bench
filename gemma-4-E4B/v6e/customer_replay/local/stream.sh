#!/usr/bin/env bash
# Stream safechat-100k from GCS straight into the pod's tmpfs, gated: corp identity check first,
# READY marker only after the in-pod row count is exactly 100000. Run from the repo root.
set -euo pipefail
NS="${NS:-rbx-bench}"
JOB="${JOB:-rbx-replay-v6e}"
EXPECT_ROWS="${EXPECT_ROWS:-100000}"

TOKEN=$(CLOUDSDK_METRICS_ENVIRONMENT=datacloud.jetski gcloud auth print-access-token)
EMAIL=$(curl -s "https://oauth2.googleapis.com/tokeninfo?access_token=${TOKEN}" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("email",""))')
ACTIVE=$(CLOUDSDK_METRICS_ENVIRONMENT=datacloud.jetski gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [ "${EMAIL}" != "ericehanley@google.com" ]; then
  # tokeninfo cannot introspect bound tokens ("invalid_token"); fall back to the active account
  # plus a real read of the bucket, which only the corp identity is allowed to do.
  if [ "${ACTIVE}" != "ericehanley@google.com" ]; then
    echo "active gcloud account is '${ACTIVE}', not the corp identity; run: gcloud auth login ericehanley@google.com --force" >&2
    exit 1
  fi
  if ! CLOUDSDK_METRICS_ENVIRONMENT=datacloud.jetski gcloud storage ls gs://gdm-rbx-text-safety-collab-sdbx/tmp/safechat-100k/part-00000.parquet > /dev/null 2>&1; then
    echo "active account ${ACTIVE} cannot read the bucket; check the identity (tokeninfo said: '${EMAIL:-no email}')" >&2
    exit 1
  fi
  echo "identity ok: ${ACTIVE} (active account; bucket readable; tokeninfo could not introspect the token)"
else
  echo "identity ok: ${EMAIL}"
fi

POD=$(kubectl -n "${NS}" get pods -l job-name="${JOB}" -o jsonpath='{.items[0].metadata.name}')
[ -n "${POD}" ] || { echo "no pod for job ${JOB}" >&2; exit 1; }
echo "pod: ${POD} on $(kubectl -n "${NS}" get pod "${POD}" -o jsonpath='{.spec.nodeName}')"

kubectl -n "${NS}" exec "${POD}" -- sh -c 'rm -f /data/READY /data/safechat.jsonl'
echo "streaming (this takes a few minutes; nothing touches local disk)..."
venv/bin/python gemma-4-E4B/g4/customer_replay/stream_dataset.py | gzip -1 \
  | kubectl -n "${NS}" exec -i "${POD}" -- sh -c 'gunzip > /data/safechat.jsonl'

ROWS=$(kubectl -n "${NS}" exec "${POD}" -- sh -c 'wc -l < /data/safechat.jsonl' | tr -d ' \r\n')
if [ "${ROWS}" != "${EXPECT_ROWS}" ]; then
  echo "row count ${ROWS} != ${EXPECT_ROWS}; READY NOT set. Re-run this script." >&2
  exit 1
fi
kubectl -n "${NS}" exec "${POD}" -- touch /data/READY
echo "READY set (${ROWS} rows in tmpfs)"
