FROM python:3.11-slim


WORKDIR /app

# System-Dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy core modules
COPY core /app/core

# Copy requirements and install
COPY agents/watchdog/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agents/watchdog/watchdog_agent.py .

# Create necessary directories
RUN mkdir -p /data /logs

CMD ["python", "watchdog_agent.py"]
