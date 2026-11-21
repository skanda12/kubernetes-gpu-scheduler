# GPU (default):  docker build -t gpuorch/inference:dev workloads/inference
# CPU variant:    docker build --build-arg BASE_IMAGE=python:3.11-slim --build-arg TORCH_CPU=1 -t gpuorch/inference:cpu workloads/inference
ARG BASE_IMAGE=pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

FROM ${BASE_IMAGE} AS deps
ARG TORCH_CPU=0
RUN pip install --no-cache-dir --target=/deps fastapi==0.111.0 uvicorn==0.30.1 prometheus_client==0.20.0 \
        transformers==4.41.2 \
 && if [ "$TORCH_CPU" = "1" ]; then pip install --no-cache-dir --target=/deps \
        torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu; fi

FROM ${BASE_IMAGE}
COPY --from=deps /deps /deps
ENV PYTHONPATH=/deps DEVICE=auto MODEL=resnet50
WORKDIR /app
COPY server.py .
EXPOSE 8000
CMD ["python", "-u", "server.py"]
