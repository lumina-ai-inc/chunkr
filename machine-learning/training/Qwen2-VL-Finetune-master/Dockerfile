FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN pip3 install uv

# Set up working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Install Python package build dependencies
RUN pip3 install setuptools wheel packaging

# Install torch first (needed for flash-attn build)
RUN uv pip install torch>=2.6.0 --system

# Install flash-attn build dependencies
RUN uv pip install ninja packaging --system

# Install flash-attn separately with build isolation disabled
RUN uv pip install flash-attn --no-build-isolation --system

# Install rest of dependencies
RUN uv pip install -e . --system

# Port for TensorBoard
EXPOSE 6006

# Simple command that will use the HF_TOKEN env var if available
RUN echo "if [ ! -z \"\$HF_TOKEN\" ]; then huggingface-cli login --token \$HF_TOKEN; fi" >> ~/.bashrc
