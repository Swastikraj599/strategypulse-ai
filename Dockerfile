FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# Install torch CPU-only first (smaller, HF Spaces doesn't need GPU for inference)
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY dashboard.html .

# HuggingFace Spaces runs on port 7860
ENV PORT=7860

# Pre-download models at build time so startup is fast
RUN python -c "\
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification; \
from sentence_transformers import SentenceTransformer; \
print('Downloading bart-large-mnli...'); \
pipeline('zero-shot-classification', model='facebook/bart-large-mnli'); \
print('Downloading finbert...'); \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
print('Downloading sentence-transformer...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('All models cached.')"

# Expose port
EXPOSE 7860

# Start the app
CMD ["python", "app.py"]
