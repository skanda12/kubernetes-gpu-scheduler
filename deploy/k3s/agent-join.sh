#!/usr/bin/env bash
# k3s AGENT inside WSL2 on a GPU laptop. Requires: systemd enabled in WSL2, NVIDIA Container Toolkit installed
# (see docs/windows-gpu-node-setup.md). k3s auto-detects nvidia-container-runtime and registers an `nvidia`
# containerd runtime; we add the RuntimeClass + device plugin from the control plane (gpu-setup.sh).
#   K3S_URL=https://<server-ip>:6443 K3S_TOKEN=<token> NODE_IP=<this-ip> bash deploy/k3s/agent-join.sh
# TODO(verify-on-gpu)
set -euo pipefail
: "${K3S_URL:?}" "${K3S_TOKEN:?}"
NODE_IP="${NODE_IP:?set NODE_IP to the LAN (mirrored) or Tailscale IP}"
IFACE="${FLANNEL_IFACE:-}"
K3S_VERSION="${K3S_VERSION:-v1.30.2+k3s1}"
NAME="${NODE_NAME:-gpu-$(hostname | tr 'A-Z' 'a-z')}"

command -v nvidia-ctk >/dev/null || { echo "install nvidia-container-toolkit first"; exit 1; }
[ -d /run/systemd/system ] || { echo "systemd is not running in WSL2: add [boot] systemd=true to /etc/wsl.conf, then 'wsl --shutdown'"; exit 1; }

ARGS=(agent --node-ip "$NODE_IP" --node-name "$NAME"
      --node-label gpuorch.io/gpu-node=true --node-label gpuorch.io/real-gpu=true --node-label gpuorch.io/wsl2=true)
[ -n "$IFACE" ] && ARGS+=(--flannel-iface "$IFACE")

curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" K3S_URL="$K3S_URL" K3S_TOKEN="$K3S_TOKEN" sh -s - "${ARGS[@]}"

sleep 5
if sudo grep -q 'nvidia' /var/lib/rancher/k3s/agent/etc/containerd/config.toml; then
  echo "OK: k3s containerd has the nvidia runtime"
else
  echo "WARN: nvidia runtime not detected by k3s; run 'sudo nvidia-ctk runtime configure --runtime=containerd' and restart k3s-agent"
fi
echo "Joined as $NAME. On the control plane run: bash deploy/k3s/gpu-setup.sh"
