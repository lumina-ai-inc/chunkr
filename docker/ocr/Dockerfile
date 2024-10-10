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

COPY services/ocr/src /app/src
COPY services/ocr/.python-version .
COPY services/ocr/download_models.py .
COPY services/ocr/pyproject.toml .
COPY services/ocr/main.py .
COPY services/ocr/uv.lock .

RUN rm -rf /app/src/__pycache__

RUN uv add paddlepaddle-gpu>=2.6.2

RUN uv run download_models.py || true

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "180"]