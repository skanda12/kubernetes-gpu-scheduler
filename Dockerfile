# One image for all control components: gpu-agent, gpu-scheduler, preemption-controller, autoscaler-sim.
#   docker build -t gpuorch/gpuorch:dev .                # CPU / mock nodes
#   docker build --build-arg EXTRAS=gpu -t gpuorch/gpuorch:dev-gpu .   # + nvidia-ml-py for NVML nodes
FROM python:3.11-slim AS build
ARG EXTRAS=""
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".${EXTRAS:+[$EXTRAS]}"

FROM python:3.11-slim
COPY --from=build /install /usr/local
COPY scenarios /etc/gpuorch/scenarios
RUN useradd -u 10001 -m gpuorch
# The agent runs as root in real (cgroup) mode via securityContext; other components run as 10001.
USER 10001
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m"]
CMD ["gpuorch.scheduler.main"]
