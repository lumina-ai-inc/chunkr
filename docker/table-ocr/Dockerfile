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

COPY services/table-ocr/.python-version .
COPY services/table-ocr/pyproject.toml .
COPY services/table-ocr/uv.lock .
COPY services/table-ocr/__init__.py .
COPY services/table-ocr/config.py .
COPY services/table-ocr/main.py .
COPY services/table-ocr/model_config.py .
COPY services/table-ocr/ocr_service.py .
COPY services/table-ocr/table_structure.py .
COPY services/table-ocr/utils.py .
RUN uv run __init__.py || true
RUN rm -rf /app/src/__pycache__

RUN wget -P /microsoft/table-transformer-structure-recognition-v1.1-all https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all/resolve/main/config.json
RUN wget -P /microsoft/table-transformer-structure-recognition-v1.1-all https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all/resolve/main/model.safetensors

EXPOSE 8000

CMD ["uv", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]