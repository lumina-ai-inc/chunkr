FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y poppler-utils ffmpeg libgomp1 libsm6 libxext6 curl unzip
RUN apt-get install -y libreoffice

# Set up CUDA environment variables
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Login to AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

WORKDIR /app

COPY services/task/src /app/src
COPY services/task/__init__.py .
COPY services/task/.python-version .
COPY services/task/pyproject.toml .
COPY services/task/main.py .
COPY services/task/uv.lock .

RUN rm -rf /app/src/__pycache__

RUN uv run __init__.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "600"]