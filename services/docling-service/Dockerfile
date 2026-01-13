FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Docling
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Docling with OCR support
RUN pip install --no-cache-dir \
    docling[ocr] \
    fastapi \
    uvicorn \
    python-multipart

# Optional: Enable GPU support (only installed if GPU available)
# RUN pip install docling[gpu]

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


