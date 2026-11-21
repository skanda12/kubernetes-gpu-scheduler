#!/usr/bin/env bash
# GPU laptop preflight (PRD §9). Run INSIDE WSL2 on a friend's laptop:
#   bash scripts/check_gpu_node.sh [CONTROL_PLANE_IP]
# Prints PASS/FAIL/WARN per check. Read-only: changes nothing.
# TODO(verify-on-gpu): run on laptops A and B and paste the output into the report.
set -u
CP_IP="${1:-${K3S_SERVER_IP:-}}"
K3S_PORT="${K3S_PORT:-6443}"
FAILS=0
pass() { printf '  \033[32mPASS\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAILS=$((FAILS+1)); }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
info() { printf '       %s\n' "$*"; }
hdr()  { printf '\n[%s] %s\n' "$1" "$2"; }

hdr 1 "WSL2 + distro"
if grep -qi microsoft /proc/version 2>/dev/null; then pass "running under WSL2 ($(uname -r))"; else fail "not WSL2 (/proc/version has no 'microsoft')"; fi
if [ -r /etc/os-release ]; then
  . /etc/os-release
  major="${VERSION_ID%%.*}"
  if [ "${ID:-}" = "ubuntu" ] && [ "${major:-0}" -ge 22 ]; then pass "$PRETTY_NAME"; else fail "need Ubuntu 22.04+, found ${PRETTY_NAME:-unknown}"; fi
fi
if ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so* >/dev/null 2>&1 && dpkg -l 2>/dev/null | grep -q '^ii  nvidia-driver'; then
  warn "a Linux nvidia-driver package is installed inside WSL2 - REMOVE it; the Windows driver provides the GPU"
fi

hdr 2 "nvidia-smi (driver from Windows)"
SMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"
if [ -x "$SMI" ] && "$SMI" >/dev/null 2>&1; then
  pass "nvidia-smi works ($SMI)"
else
  fail "nvidia-smi failed - update the Windows NVIDIA driver (>= 535) and run 'wsl --update' in PowerShell"
fi

hdr 3 "NVML from Python (pynvml)"
PYOUT="$(python3 - <<'EOF' 2>&1
try:
    import pynvml
except ImportError:
    print("NOIMPORT"); raise SystemExit
pynvml.nvmlInit()
n = pynvml.nvmlDeviceGetCount()
print(f"count={n}")
for i in range(n):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    try:
        u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    except pynvml.NVMLError as e:
        u = f"ERR({e})"
    print(f"gpu{i} mem_total={m.total//2**20}MiB mem_used={m.used//2**20}MiB util={u}")
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        print(f"per_process=supported({len(procs)} procs visible)")
    except pynvml.NVMLError as e:
        print(f"per_process=unsupported({e})")
EOF
)"
if echo "$PYOUT" | grep -q NOIMPORT; then
  fail "pynvml missing: pip install nvidia-ml-py"
elif echo "$PYOUT" | grep -q "count="; then
  pass "NVML readable"; echo "$PYOUT" | sed 's/^/       /'
  echo "$PYOUT" | grep -q "per_process=supported" || warn "per-process NVML not available (expected on WSL2; gpuorch uses workload self-reporting)"
else
  fail "NVML init failed:"; echo "$PYOUT" | sed 's/^/       /'
fi

hdr 4 "container runtime + NVIDIA Container Toolkit"
if command -v docker >/dev/null 2>&1; then pass "docker: $(docker --version 2>/dev/null)"; else info "docker not installed (fine if using k3s containerd)"; fi
if command -v containerd >/dev/null 2>&1 || [ -x /usr/local/bin/k3s ]; then pass "containerd/k3s present"; else warn "no containerd/k3s yet (install k3s agent per docs/windows-gpu-node-setup.md)"; fi
if command -v nvidia-ctk >/dev/null 2>&1; then pass "nvidia-container-toolkit: $(nvidia-ctk --version 2>/dev/null | head -1)"; else fail "nvidia-container-toolkit not installed"; fi
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  if timeout 120 docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi -L >/tmp/gpuorch-docker-gpu.txt 2>&1; then
    pass "docker --gpus all works: $(head -1 /tmp/gpuorch-docker-gpu.txt)"
  else
    fail "docker --gpus all failed: $(tail -2 /tmp/gpuorch-docker-gpu.txt)"
  fi
fi

hdr 5 "cgroup v2 + freezer"
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
  pass "cgroup v2 mounted (controllers: $(cat /sys/fs/cgroup/cgroup.controllers))"
  if [ -e /sys/fs/cgroup/cgroup.freeze ] || find /sys/fs/cgroup -maxdepth 2 -name cgroup.freeze -print -quit 2>/dev/null | grep -q .; then
    pass "cgroup.freeze available (throttle = freeze works)"
  else
    fail "cgroup.freeze not found"
  fi
else
  fail "cgroup v2 not mounted - add to %UserProfile%\\.wslconfig: [wsl2] kernelCommandLine = cgroup_no_v1=all"
fi

hdr 6 "networking"
MODE="NAT"
if ip -4 addr show 2>/dev/null | grep -qE 'inet (192\.168|10\.|172\.(1[6-9]|2[0-9]|3[01]))' && ! ip -4 addr show eth0 2>/dev/null | grep -q '172\.'; then MODE="mirrored (likely)"; fi
if ip link show tailscale0 >/dev/null 2>&1; then MODE="$MODE + tailscale ($(ip -4 -o addr show tailscale0 | awk '{print $4}'))"; fi
LAN_IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')"
info "networking mode: $MODE"
info "primary IP: ${LAN_IP:-unknown}"
if [ -n "$CP_IP" ]; then
  if timeout 3 bash -c "</dev/tcp/$CP_IP/$K3S_PORT" 2>/dev/null; then pass "k3s server $CP_IP:$K3S_PORT reachable"; else fail "cannot reach $CP_IP:$K3S_PORT (firewall? mirrored networking? same Wi-Fi?)"; fi
else
  warn "pass the control-plane IP to test reachability: bash scripts/check_gpu_node.sh <IP>"
fi

hdr 7 "report info"
if [ -x "$SMI" ]; then
  "$SMI" --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/       GPU: /'
  "$SMI" 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | sed 's/^/       /'
fi
info "kernel: $(uname -r)"

printf '\n%s\n' "$( [ $FAILS -eq 0 ] && echo 'ALL CHECKS PASSED' || echo "$FAILS CHECK(S) FAILED" )"
exit $FAILS
