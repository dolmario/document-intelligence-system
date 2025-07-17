FROM python:3.11-slim

WORKDIR /app

ENV PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY core /app/core

COPY agents/indexer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download de_core_news_sm

COPY agents/indexer /app/agents/indexer

RUN mkdir -p /indices /app/logs
RUN chmod -R 755 /app/logs /indices

CMD ["python", "-m", "agents.indexer.indexer_agent"]
