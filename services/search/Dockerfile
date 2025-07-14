FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy core modules first
COPY core /app/core
COPY services /app/services

# Install requirements
COPY services/search/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download de_core_news_sm

# Create directories
RUN mkdir -p /indices /app/logs

# Set permissions
RUN chmod -R 755 /app/logs

CMD ["python", "-m", "uvicorn", "services.search.api:app", "--host", "0.0.0.0", "--port", "8001"]
