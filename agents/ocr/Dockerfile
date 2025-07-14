FROM python:3.11-slim

WORKDIR /app

# System-Dependencies für OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    poppler-utils \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy core modules
COPY core /app/core

# Copy requirements and install
COPY agents/ocr/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agents/ocr/ocr_agent.py .

# Create necessary directories
RUN mkdir -p /data /indices /app/logs

# Set permissions
RUN chmod -R 755 /app/logs

CMD ["python", "ocr_agent.py"]
