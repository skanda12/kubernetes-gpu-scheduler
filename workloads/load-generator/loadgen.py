"""Open-loop load generator.

Env: TARGET_URL=http://inference:8000/predict  RPS=20  DURATION_S=300  SLO_MS=50  PORT=8001  OUT_CSV=
Sends requests at a fixed rate (open loop: slow responses do not reduce the offered load), records latency
(client-side, includes queueing), exposes `loadgen_request_latency_seconds` on :8001/metrics, and prints a JSON
summary (p50/p95/p99, SLO violation %) at the end.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time

import httpx
from prometheus_client import Counter, Histogram, start_http_server

URL = os.environ.get("TARGET_URL", "http://inference:8000/predict")
RPS = float(os.environ.get("RPS", "20"))
DURATION = float(os.environ.get("DURATION_S", "300"))
SLO_MS = float(os.environ.get("SLO_MS", "0") or 0)
OUT_CSV = os.environ.get("OUT_CSV", "")
TARGET = os.environ.get("TARGET_NAME", URL)

LAT = Histogram("loadgen_request_latency_seconds", "client latency", ["target"],
                buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 5))
ERR = Counter("loadgen_errors_total", "failed requests", ["target"])


def pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(q / 100 * len(s)))], 2)


async def one(client: httpx.AsyncClient, results: list) -> None:
    t0 = time.perf_counter()
    ok = True
    try:
        r = await client.post(URL, timeout=10)
        ok = r.status_code == 200
    except httpx.HTTPError:
        ok = False
    ms = (time.perf_counter() - t0) * 1000
    if ok:
        LAT.labels(TARGET).observe(ms / 1000)
    else:
        ERR.labels(TARGET).inc()
    results.append((time.time(), ms, ok))


async def main() -> None:
    start_http_server(int(os.environ.get("PORT", "8001")))
    results: list[tuple[float, float, bool]] = []
    interval = 1.0 / RPS
    async with httpx.AsyncClient() as client:
        tasks = []
        t_start = time.perf_counter()
        n = 0
        while time.perf_counter() - t_start < DURATION:
            tasks.append(asyncio.create_task(one(client, results)))
            n += 1
            await asyncio.sleep(max(0.0, t_start + n * interval - time.perf_counter()))
        await asyncio.gather(*tasks)
    lats = [ms for _, ms, ok in results if ok]
    errors = sum(1 for *_, ok in results if not ok)
    viol = sum(1 for _, ms, ok in results if not ok or (SLO_MS and ms > SLO_MS))
    summary = {"target": TARGET, "requests": len(results), "errors": errors, "p50_ms": pct(lats, 50),
               "p95_ms": pct(lats, 95), "p99_ms": pct(lats, 99),
               "slo_violation_pct": round(100 * viol / len(results), 2) if results and SLO_MS else None}
    print(json.dumps(summary), flush=True)
    if OUT_CSV:
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ts", "latency_ms", "ok"])
            w.writerows(results)


if __name__ == "__main__":
    asyncio.run(main())
