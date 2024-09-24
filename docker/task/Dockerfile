FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    htop \
    build-essential \
    dkms \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY services/task .

RUN uv pip install .[linux]

EXPOSE 3000

CMD ["uv", "run", "bentoml", "serve", "service:Task"]