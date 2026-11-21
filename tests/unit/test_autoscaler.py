from __future__ import annotations

from gpuorch.autoscaler_sim.core import AutoscalerConfig, AutoscalerSim
from gpuorch.common import config as C
from gpuorch.common.models import NodeGpuSummary
from tests.conftest import gpu_node, gpu_pod


def test_scale_up_down_cost_and_kwok_reaper(cluster, clock):
    cluster.add_node(gpu_node("w0", slots=1, ts=clock()))
    cfg = AutoscalerConfig(scale_up_after_s=10, scale_down_after_s=30, scale_up_cooldown_s=5, max_sim_nodes=1,
                           price_per_hour=3.6)
    asim = AutoscalerSim(cluster, cfg, clock)
    cluster.add_pod(gpu_pod("p", uid="p", created=clock(), annotations={C.ANN_DURATION_S: "20", C.ANN_GPU_MEM_MB: "1000"}))
    asim.run_once()
    assert len(cluster.nodes) == 1  # too young
    clock.advance(11)
    asim.run_once()
    sim = [n for n in cluster.nodes.values() if n.simulated]
    assert len(sim) == 1 and sim[0].alloc_gpu == 4 and sim[0].taints[0].key == "kwok.x-k8s.io/node"
    clock.advance(10)
    asim.run_once()
    assert len([n for n in cluster.nodes.values() if n.simulated]) == 1  # max_sim_nodes

    pod = cluster.get_pod("p")
    pod.node_name, pod.phase, pod.start_time = sim[0].name, "Running", clock()
    asim.run_once()
    st = NodeGpuSummary.from_annotation(cluster.nodes[sim[0].name].annotations[C.ANN_NODE_STATE])
    assert st.mem_used_mb == 1300 and st.util == 50
    clock.advance(21)
    asim.run_once()
    assert cluster.get_pod("p").phase == "Succeeded"
    clock.advance(1)
    asim.run_once()  # first cycle that sees the node empty starts the idle timer
    assert any(n.simulated for n in cluster.nodes.values())
    clock.advance(31)
    asim.run_once()
    assert not any(n.simulated for n in cluster.nodes.values())
    # cost: $3.6/h == $0.001/s per GPU node
    assert 0.05 < asim.cost_dollars < 0.2
    assert asim.node_seconds > 0
