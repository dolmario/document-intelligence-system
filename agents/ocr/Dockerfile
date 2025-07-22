FROM python:3.11-slim

WORKDIR /app

# System Dependencies für OCR V2 + GPU + OpenCV
RUN apt-get update && apt-get install -y \
    # OCR
    tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng \
    # PDF Processing  
    poppler-utils \
    # Build tools
    gcc libpq-dev python3-dev \
    # OpenCV + GPU Dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    # Media Libraries (ffmpeg bringt die libav* mit)
    ffmpeg \
    # GPU Support (falls verfügbar)
    wget gnupg2 \
    # Zusätzlich für OCR
    libfontconfig1 \
    libice6 \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies (inklusive GPU Support)
COPY agents/ocr/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY agents/ocr/ocr_agent.py ocr_agent.py

# Directories
RUN mkdir -p /app/logs

# Environment (nur die wichtigsten)
ENV PYTHONUNBUFFERED=1 \
    TESSERACT_LANG=deu+eng \
    CHUNK_SIZE=1000 \
    OPENCV_IO_ENABLE_OPENEXR=1

CMD ["python", "-u", "ocr_agent.py"]