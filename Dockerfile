FROM python:3.11-slim

# System deps for PyMuPDF / pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY evaluation/ ./evaluation/
COPY frontend/ ./frontend/

# Pre-create storage directories
RUN mkdir -p storage/uploads storage/parsed storage/chunked storage/chroma \
             evaluation/results

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
