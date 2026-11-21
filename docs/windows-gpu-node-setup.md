# Turning a Windows GPU laptop into a gpuorch worker (WSL2)

For friends lending laptops A and B. **No dual-boot, nothing installed into Windows except WSL2 and an updated
NVIDIA driver.** Everything else lives inside the WSL2 Ubuntu and can be removed afterwards (step 10).

> Never install a Linux NVIDIA driver inside WSL2. The Windows driver provides the GPU to Linux.

## 0. What you need

- Windows 11 22H2+ (Windows 10 works only with option 2 networking: Tailscale)
- NVIDIA GPU, ~30 GB free disk, admin rights
- All laptops on the same Wi-Fi / phone hotspot (option 1) **or** a free Tailscale account (option 2)

## 1. Update the Windows NVIDIA driver

Install the latest Game Ready or Studio driver from nvidia.com (>= 535). Reboot.

## 2. Install WSL2 + Ubuntu 22.04 (PowerShell as admin)

```powershell
wsl --install -d Ubuntu-22.04
wsl --update
```

Reboot if asked, open "Ubuntu 22.04", create a user.

## 3. Configure WSL2 (`%UserProfile%\.wslconfig`)

Copy `deploy/k3s/wslconfig.example` to `%UserProfile%\.wslconfig`. It enables **mirrored networking**
(WSL2 gets the laptop's LAN IP) and a pure cgroup v2 hierarchy (needed for throttle = freeze). Then:

```powershell
wsl --shutdown
```

Inside Ubuntu, enable systemd (k3s needs it):

```bash
printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
# back in PowerShell: wsl --shutdown, then reopen Ubuntu
```

## 4. Open the firewall (PowerShell as admin)

```powershell
cd <repo>\deploy\k3s
powershell -ExecutionPolicy Bypass -File .\firewall.ps1
```

## 5. NVIDIA Container Toolkit (inside Ubuntu)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit python3-pip
pip install nvidia-ml-py
```

## 6. Preflight

```bash
git clone <repo> && cd gpu-orchestrator
bash scripts/check_gpu_node.sh <DEV_LAPTOP_IP>
```

All checks should PASS. The script prints the GPU model, VRAM, driver and CUDA version for the report. A WARN about
per-process NVML is expected on WSL2: gpuorch uses workload self-reporting instead.

## 7. Join the cluster

On the **dev laptop** (control plane, CPU only):

```bash
NODE_IP=<dev-laptop-ip> bash deploy/k3s/server-install.sh   # prints the join token
```

On **each GPU laptop**:

```bash
K3S_URL=https://<dev-laptop-ip>:6443 K3S_TOKEN=<token> NODE_IP=<this-laptop-ip> bash deploy/k3s/agent-join.sh
```

### Networking option 2: Tailscale (if mirrored mode or the Wi-Fi blocks peer traffic)

Inside WSL2 on every laptop: `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`.
Use the `100.x.y.z` addresses as `NODE_IP` and add `FLANNEL_IFACE=tailscale0` to both install commands.

### Fallback if multi-node networking fails

Run k3s **server + agent on one GPU laptop** (single real GPU node) and let `autoscaler-sim` add KWOK nodes for
the rest. Say so clearly in the report: "1 real GPU node + N simulated nodes".

## 8. GPU plumbing + gpuorch (dev laptop)

```bash
make images-gpu   # then make the images available on every node (registry or `k3s ctr images import`)
bash deploy/k3s/gpu-setup.sh
```

**Accept (M5):** `kubectl get nodes` shows both laptops Ready; the smoke-test Job prints `nvidia-smi` output and
`cuda True <GPU name>`; `kubectl -n gpuorch port-forward ds/gpu-agent-real 9400` then `curl localhost:9400/state`
shows real memory/util; throttling a training pod (`curl -XPOST localhost:9400/throttle -H 'content-type: application/json' -d '{"pod_uid":"<uid>"}'`)
drops its GPU utilisation in `nvidia-smi` (watch with `nvidia-smi dmon -s u`).

## 9. Run the evaluation (M6)

See `docs/gpu-evaluation.md`.

## 10. Uninstall cleanly afterwards

```bash
bash deploy/k3s/uninstall.sh        # inside WSL2
```

```powershell
powershell -ExecutionPolicy Bypass -File deploy\k3s\firewall.ps1 -Remove   # admin PowerShell
del %UserProfile%\.wslconfig        # or restore your previous one
wsl --unregister Ubuntu-22.04       # optional: deletes the whole Linux install
```
