FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip

# Install PyTorch
RUN pip3 install torch==2.3.1 torchvision torchaudio

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y poppler-utils ffmpeg libgomp1 libsm6 libxext6

# Set up CUDA environment variables
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

COPY services/task .

RUN uv run __init__.py

RUN uv pip install .[linux]

RUN uv run download_models.py || true

EXPOSE 3000

CMD ["uv", "run", "bentoml", "serve", "service:Task"]