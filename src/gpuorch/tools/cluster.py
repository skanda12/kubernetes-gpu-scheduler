"""Local cluster lifecycle (cross-platform; the Makefile calls these).

    python -m gpuorch.tools.cluster check-tools
    python -m gpuorch.tools.cluster up        # kind: 1 control-plane + 3 workers, fake nvidia.com/gpu, KWOK
    python -m gpuorch.tools.cluster images    # docker build gpuorch + fake workload, kind load
    python -m gpuorch.tools.cluster deploy    # helm upgrade --install + grafana dashboards ConfigMap
    python -m gpuorch.tools.cluster status
    python -m gpuorch.tools.cluster down
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLUSTER = os.environ.get("KIND_CLUSTER", "gpuorch")
NAMESPACE = os.environ.get("GPUORCH_NAMESPACE", "gpuorch")
FAKE_GPUS_PER_WORKER = int(os.environ.get("FAKE_GPUS_PER_WORKER", "4"))  # 1 GPU x 4 time-slicing replicas
KWOK_VERSION = os.environ.get("KWOK_VERSION", "v0.6.1")
IMAGE = os.environ.get("GPUORCH_IMAGE", "gpuorch/gpuorch:dev")
FAKE_IMAGE = os.environ.get("FAKE_IMAGE", "gpuorch/fake-workload:dev")

INSTALL = {
    "Windows": {
        "docker": "winget install -e --id Docker.DockerDesktop   (then start Docker Desktop; enable WSL2 backend)",
        "kind": "winget install -e --id Kubernetes.kind",
        "kubectl": "winget install -e --id Kubernetes.kubectl",
        "helm": "winget install -e --id Helm.Helm",
        "make": "optional: winget install -e --id GnuWin32.Make  (or use `python -m gpuorch.tools.cluster ...`)",
    },
    "Darwin": {
        "docker": "brew install --cask docker", "kind": "brew install kind", "kubectl": "brew install kubectl",
        "helm": "brew install helm", "make": "xcode-select --install",
    },
    "Linux": {
        "docker": "https://docs.docker.com/engine/install/ubuntu/  (then: sudo usermod -aG docker $USER)",
        "kind": "go install sigs.k8s.io/kind@v0.23.0  or  curl -Lo kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64",
        "kubectl": "sudo snap install kubectl --classic",
        "helm": "curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash",
        "make": "sudo apt install make",
    },
}


def run(cmd: list[str], check: bool = True, capture: bool = False, input: str | None = None) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture, input=input)


def check_tools(required: tuple[str, ...] = ("docker", "kind", "kubectl", "helm")) -> bool:
    osname = platform.system()
    ok = True
    for tool in (*required, "make"):
        found = shutil.which(tool)
        status = "ok" if found else ("MISSING" if tool in required else "missing (optional)")
        print(f"  {tool:8s} {status}{'  ' + found if found else ''}")
        if not found:
            ok = ok and tool not in required
            print(f"           install: {INSTALL.get(osname, INSTALL['Linux']).get(tool, '?')}")
    if shutil.which("docker"):
        r = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if r.returncode != 0:
            print("  docker daemon not reachable: start Docker Desktop / the docker service")
            ok = False
    print(f"  python   {sys.version.split()[0]} (3.11+ required)")
    return ok


def _require() -> None:
    if not check_tools():
        sys.exit("missing tools - see install hints above")


def up() -> None:
    _require()
    existing = run(["kind", "get", "clusters"], capture=True).stdout.split()
    if CLUSTER not in existing:
        ckpt = (ROOT / ".data" / "ckpt").resolve()
        ckpt.mkdir(parents=True, exist_ok=True)
        cfg = (ROOT / "deploy" / "kind" / "cluster.yaml").read_text().replace("${CKPT_HOST_DIR}", ckpt.as_posix())
        cfg = cfg.replace("name: gpuorch", f"name: {CLUSTER}")
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(cfg)
        run(["kind", "create", "cluster", "--config", fh.name, "--wait", "120s"])
        os.unlink(fh.name)
    else:
        print(f"kind cluster {CLUSTER!r} already exists")
    advertise_fake_gpus()
    install_kwok()
    run(["kubectl", "get", "nodes", "-o",
         "custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com/gpu,READY:.status.conditions[-1].status"])


def advertise_fake_gpus() -> None:
    """Patch workers' node status with fake nvidia.com/gpu capacity (extended resource)."""
    from gpuorch.common.k8s import KubeCluster

    kube = KubeCluster("cluster-tool")
    for n in kube.v1.list_node(label_selector="gpuorch.io/gpu-node=true").items:
        kube.patch_node_gpu_capacity(n.metadata.name, FAKE_GPUS_PER_WORKER)
        print(f"  {n.metadata.name}: nvidia.com/gpu={FAKE_GPUS_PER_WORKER}")


def install_kwok() -> None:
    base = f"https://github.com/kubernetes-sigs/kwok/releases/download/{KWOK_VERSION}"
    run(["kubectl", "apply", "-f", f"{base}/kwok.yaml"])
    run(["kubectl", "apply", "-f", f"{base}/stage-fast.yaml"])
    # stage-fast's `pod-complete` marks Job pods Succeeded the moment they run. Remove it: autoscaler-sim
    # completes pods on KWOK nodes after their real gpuorch.io/duration-s instead.
    run(["kubectl", "delete", "stage", "pod-complete", "--ignore-not-found"])


def images(gpu: bool = False) -> None:
    _require()
    run(["docker", "build", "-t", IMAGE, str(ROOT)])
    run(["docker", "build", "-t", FAKE_IMAGE, str(ROOT / "workloads" / "fake")])
    if gpu:
        run(["docker", "build", "--build-arg", "EXTRAS=gpu", "-t", IMAGE.replace(":dev", ":dev-gpu"), str(ROOT)])
    for img in (IMAGE, FAKE_IMAGE):
        run(["kind", "load", "docker-image", img, "--name", CLUSTER])


def deploy(extra: list[str] | None = None) -> None:
    _require()
    run(["kubectl", "create", "namespace", NAMESPACE], check=False, capture=True)
    dash = run(["kubectl", "create", "configmap", "gpuorch-grafana-dashboards", "-n", NAMESPACE,
                f"--from-file={ROOT / 'deploy' / 'grafana'}", "--dry-run=client", "-o", "yaml"], capture=True).stdout
    run(["kubectl", "apply", "-f", "-"], input=dash)
    run(["helm", "upgrade", "--install", "gpuorch", str(ROOT / "deploy" / "helm" / "gpuorch"), "-n", NAMESPACE,
         "--wait", "--timeout", "180s", *(extra or [])])
    status()


def status() -> None:
    run(["kubectl", "get", "pods", "-n", NAMESPACE, "-o", "wide"], check=False)
    print("\nPrometheus: http://localhost:30090 (kind: `kubectl -n gpuorch port-forward svc/prometheus 9090`)")
    print("Grafana:    `kubectl -n gpuorch port-forward svc/grafana 3000` -> http://localhost:3000 (admin/gpuorch)")


def down() -> None:
    run(["kind", "delete", "cluster", "--name", CLUSTER])


def wait_for(predicate, timeout: float, interval: float = 1.0, what: str = "condition") -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if predicate():
            return time.time() - t0
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-tools")
    sub.add_parser("up")
    sub.add_parser("down")
    sub.add_parser("status")
    im = sub.add_parser("images")
    im.add_argument("--gpu", action="store_true")
    dp = sub.add_parser("deploy")
    dp.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    if a.cmd == "check-tools":
        sys.exit(0 if check_tools() else 1)
    if a.cmd == "deploy":
        deploy([x for s in a.set for x in ("--set", s)])
    elif a.cmd == "images":
        images(a.gpu)
    else:
        {"up": up, "down": down, "status": status}[a.cmd]()


if __name__ == "__main__":
    main()
