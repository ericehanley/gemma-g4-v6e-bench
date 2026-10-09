#!/usr/bin/env bash
# Apply everything for the v6e replay in the right order (namespace/SA -> secret -> ConfigMap -> Job),
# so the Job never starts before its mounts exist. Run from the repo root.
set -euo pipefail
NS=rbx-bench
DIR=gemma-4-E4B/v6e/customer_replay
MANIFEST="${MANIFEST:-${DIR}/rbx-replay-v6e.yaml}"

echo "context: $(kubectl config current-context)"
kubectl get ns "${NS}" > /dev/null 2>&1 || kubectl create namespace "${NS}"
kubectl label ns "${NS}" purpose=roblox-gemma4-benchmark --overwrite > /dev/null
kubectl -n "${NS}" get sa rbx-bench > /dev/null 2>&1 || kubectl -n "${NS}" apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rbx-bench
automountServiceAccountToken: false
EOF

if ! kubectl -n "${NS}" get secret hf-token > /dev/null 2>&1; then
  kubectl -n "${NS}" create secret generic hf-token \
    --from-literal=HF_TOKEN="$(kubectl -n default get secret hf-token -o jsonpath='{.data.HF_TOKEN}' | base64 -d)"
fi

# Client scripts come unchanged from the G4 run; only the driver is TPU-specific.
kubectl -n "${NS}" create configmap rbx-replay-v6e-scripts \
  --from-file=gemma-4-E4B/g4/customer_replay/scripts/ \
  --from-file="${DIR}/scripts/" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "${MANIFEST}"
kubectl -n "${NS}" get job,pods -o wide
