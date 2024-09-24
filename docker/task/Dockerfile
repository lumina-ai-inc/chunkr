FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN python --version

# Install system dependencies and tools
RUN apt-get update && apt-get install -y \
    htop \
    build-essential \
    dkms \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

WORKDIR /app

COPY services/task .

RUN uv run pip install .[linux]

EXPOSE 3000

CMD ["uv", "run", "bentoml", "serve", "service:Task"]

