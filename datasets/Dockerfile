FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV MODEL_PATH=/app/outputs/checkpoints/model_final.pt
ENV TOKENIZER_PATH=/app/outputs/tokenizer/tokenizer.json
ENV CONFIG_PATH=/app/outputs/flask_deployment/config.json
ENV DEVICE=cuda
ENV BATCH_SIZE=8
ENV MAX_CACHE_SIZE=1000
ENV MAX_SEQUENCE_LENGTH=1024
ENV PYTHONPATH=/app

# Create WSGI entry point
RUN echo 'import os\n\
import torch\n\
import json\n\
import sys\n\
sys.path.append("/app")\n\
from src.deployment.flask_server import app, ModelServer\n\
\n\
# Initialize model server\n\
model_server = ModelServer(\n\
    model_path=os.environ.get("MODEL_PATH"),\n\
    tokenizer_path=os.environ.get("TOKENIZER_PATH"),\n\
    config_path=os.environ.get("CONFIG_PATH"),\n\
    device=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),\n\
    batch_size=int(os.environ.get("BATCH_SIZE", 8)),\n\
    max_cache_size=int(os.environ.get("MAX_CACHE_SIZE", 1000)),\n\
    max_sequence_length=int(os.environ.get("MAX_SEQUENCE_LENGTH", 1024))\n\
)\n\
\n\
if __name__ == "__main__":\n\
    app.run()\n\
' > /app/wsgi.py

# Expose port
EXPOSE 5000

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "wsgi:app"]