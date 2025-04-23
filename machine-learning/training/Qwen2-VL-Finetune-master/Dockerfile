FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app
COPY . /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git wget python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install uv
RUN pip install --no-cache-dir uv

# Create dummy torch installation to prevent uv from trying to build it
RUN mkdir -p /usr/local/lib/python3.11/dist-packages/torch

# Write stub script for flash-attn to use during installation
RUN echo 'class CUDAVersion:\n    def __init__(self): pass\n    major = 11\n    minor = 8\ndef cuda_version():\n    return CUDAVersion()\ndef __getattr__(name):\n    return None' > /usr/local/lib/python3.11/dist-packages/torch/__init__.py

# Create wrapper script to handle torch and flash-attn correctly
RUN echo '#!/bin/bash\n\
    source /app/.venv/bin/activate\n\
    exec python "$@"' > /usr/local/bin/run-with-venv && \
    chmod +x /usr/local/bin/run-with-venv

# Install all dependencies using plain pip
RUN python -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir setuptools wheel ninja packaging && \
    pip install --no-cache-dir torch==2.1.2 && \
    pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation && \
    pip install --no-cache-dir -e .

# Ensure scripts are executable
RUN chmod +x scripts/*.sh

# Set up script execution environment
RUN echo 'source /app/.venv/bin/activate' >> ~/.bashrc
RUN echo 'if [ ! -z "$HF_TOKEN" ]; then huggingface-cli login --token $HF_TOKEN; fi' >> ~/.bashrc

# Port for TensorBoard
EXPOSE 6006

# Set the entrypoint to use our venv wrapper
ENTRYPOINT ["/bin/bash", "-c"]
