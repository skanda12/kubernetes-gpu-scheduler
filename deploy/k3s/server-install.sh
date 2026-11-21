#!/usr/bin/env bash
# k3s SERVER (control plane) on the dev laptop (in WSL2 if it is a Windows laptop). CPU only.
#   NODE_IP=<lan-or-tailscale-ip> bash deploy/k3s/server-install.sh
# TODO(verify-on-gpu): run when the friend laptops are available; then `kubectl get nodes`.
set -euo pipefail
NODE_IP="${NODE_IP:?set NODE_IP to the LAN (mirrored) or Tailscale IP}"
IFACE="${FLANNEL_IFACE:-}"   # e.g. tailscale0 when using Tailscale
K3S_VERSION="${K3S_VERSION:-v1.30.2+k3s1}"

ARGS=(server --node-ip "$NODE_IP" --advertise-address "$NODE_IP" --tls-san "$NODE_IP"
      --disable traefik --write-kubeconfig-mode 644
      --node-taint node-role.kubernetes.io/control-plane=true:NoSchedule)
[ -n "$IFACE" ] && ARGS+=(--flannel-iface "$IFACE")

curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="$K3S_VERSION" sh -s - "${ARGS[@]}"

echo
echo "Join token (give this to the GPU laptops):"
sudo cat /var/lib/rancher/k3s/server/node-token
echo
echo "kubeconfig: /etc/rancher/k3s/k3s.yaml  (export KUBECONFIG=/etc/rancher/k3s/k3s.yaml)"
echo "Next on each GPU laptop: K3S_URL=https://$NODE_IP:6443 K3S_TOKEN=<token> bash deploy/k3s/agent-join.sh"
