FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends librdkafka1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY streaming/requirements.txt /app/streaming/requirements.txt
RUN pip install --no-cache-dir -r /app/streaming/requirements.txt
COPY streaming /app/streaming

ENV PYTHONPATH=/app
