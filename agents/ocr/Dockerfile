FROM python:3.11-slim

WORKDIR /app

ENV PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    poppler-utils \
    libpoppler-cpp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY core /app/core

COPY agents/ocr/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download de_core_news_sm

COPY agents/ocr/ocr_agent.py .

RUN mkdir -p /data /indices /app/logs

CMD ["python", "ocr_agent.py"]
