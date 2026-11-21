FROM python:3.11-slim AS deps
RUN pip install --no-cache-dir --target=/deps httpx==0.27.0 prometheus_client==0.20.0

FROM python:3.11-slim
COPY --from=deps /deps /deps
ENV PYTHONPATH=/deps
WORKDIR /app
COPY loadgen.py .
EXPOSE 8001
CMD ["python", "-u", "loadgen.py"]
