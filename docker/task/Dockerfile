FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update && apt-get install -y poppler-utils

WORKDIR /app

COPY services/task .

RUN uv venv

RUN uv pip install .[linux] --system

EXPOSE 3000

CMD ["uv", "run", "bentoml", "serve", "service:Task"]