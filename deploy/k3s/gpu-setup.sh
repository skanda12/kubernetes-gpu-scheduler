#!/usr/bin/env bash
# Run on the control plane after GPU agents joined: RuntimeClass, NVIDIA device plugin with time-slicing,
# smoke tests, then gpuorch with real agents on GPU nodes.
# TODO(verify-on-gpu): expected result - `kubectl describe node gpu-...` shows nvidia.com/gpu: 4
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
DP_VERSION="${DEVICE_PLUGIN_VERSION:-v0.15.0}"

kubectl apply -f "$HERE/runtimeclass.yaml"
kubectl apply -f "$HERE/device-plugin-timeslicing.yaml"
kubectl apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/$DP_VERSION/deployments/static/nvidia-device-plugin.yml" || true
# point the device plugin at our time-slicing config and the nvidia runtime class
kubectl -n kube-system patch daemonset nvidia-device-plugin-daemonset --type merge -p "$(cat "$HERE/device-plugin-patch.json")"
kubectl -n kube-system rollout status ds/nvidia-device-plugin-daemonset --timeout=180s

echo "waiting for nvidia.com/gpu capacity on real GPU nodes..."
for i in $(seq 1 60); do
  caps=$(kubectl get nodes -l gpuorch.io/real-gpu=true -o jsonpath='{range .items[*]}{.metadata.name}={.status.capacity.nvidia\.com/gpu}{"\n"}{end}')
  echo "$caps" | grep -qE '=[1-9]' && break
  sleep 5
done
echo "$caps"

echo "--- smoke test: nvidia-smi + torch.cuda in a pod (PRD M5 accept) ---"
kubectl delete -f "$HERE/gpu-smoke-test.yaml" --ignore-not-found
kubectl apply -f "$HERE/gpu-smoke-test.yaml"
kubectl wait --for=condition=complete job/gpu-smoke-test --timeout=600s && kubectl logs job/gpu-smoke-test

echo "--- deploying gpuorch with real agents on GPU nodes ---"
kubectl create namespace gpuorch --dry-run=client -o yaml | kubectl apply -f -
kubectl -n gpuorch create configmap gpuorch-grafana-dashboards --from-file="$ROOT/deploy/grafana" --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install gpuorch "$ROOT/deploy/helm/gpuorch" -n gpuorch -f "$HERE/values-k3s.yaml"
