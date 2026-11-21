# GPU (default):  docker build -t gpuorch/training:dev workloads/training
# CPU variant:    docker build --build-arg BASE_IMAGE=python:3.11-slim --build-arg TORCH_CPU=1 -t gpuorch/training:cpu workloads/training
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

FROM ${BASE_IMAGE} AS deps
ARG TORCH_CPU=0
RUN pip install --no-cache-dir --target=/deps prometheus_client==0.20.0 \
 && if [ "$TORCH_CPU" = "1" ]; then pip install --no-cache-dir --target=/deps \
        torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu; fi

FROM ${BASE_IMAGE}
COPY --from=deps /deps /deps
ENV PYTHONPATH=/deps DEVICE=auto
WORKDIR /app
COPY train.py .
EXPOSE 8000
CMD ["python", "-u", "train.py"]
