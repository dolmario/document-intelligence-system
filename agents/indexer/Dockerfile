FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy core modules
COPY core /app/core

# Copy requirements and install
COPY agents/indexer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download de_core_news_sm

# Copy agent code
COPY agents/indexer /app/agents/indexer

# Create directories
RUN mkdir -p /indices /app/logs

# Set permissions
RUN chmod -R 755 /app/logs /indices

CMD ["python", "-m", "agents.indexer.indexer_agent"]
