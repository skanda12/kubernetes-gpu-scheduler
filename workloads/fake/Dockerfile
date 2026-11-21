FROM python:3.11-slim
RUN useradd -u 10001 -m app
WORKDIR /app
COPY app.py .
USER 10001
EXPOSE 8000
ENTRYPOINT ["python", "-u", "app.py"]
