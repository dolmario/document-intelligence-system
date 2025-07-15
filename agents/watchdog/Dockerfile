FROM python:3.11-slim

WORKDIR /app

ENV PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY core /app/core

COPY agents/watchdog/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/watchdog/watchdog_agent.py .

RUN mkdir -p /data /app/logs
RUN chmod -R 755 /app/logs

CMD ["python", "watchdog_agent.py"]
