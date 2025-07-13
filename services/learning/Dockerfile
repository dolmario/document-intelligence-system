FROM python:3.11-slim

WORKDIR /app

# Copy core modules
COPY core /app/core
COPY services /app/services

# Install requirements
COPY services/learning/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "services.learning.learning_service"]
