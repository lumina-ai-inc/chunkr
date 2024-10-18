FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y poppler-utils ffmpeg libgomp1 libsm6 libxext6

# Set up CUDA environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

COPY services/rapid-ocr/.python-version .
COPY services/rapid-ocr/pyproject.toml .
COPY services/rapid-ocr/main.py .
COPY services/rapid-ocr/uv.lock .
COPY services/rapid-ocr/__init__.py .
RUN uv run __init__.py || true
RUN rm -rf /app/src/__pycache__

EXPOSE 8000

CMD ["uv", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]