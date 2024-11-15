FROM ghcr.io/mindee/doctr:tf-py3.8.18-gpu-2023-09

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN source $HOME/.cargo/env

WORKDIR /app

COPY services/doctr/.python-version .
COPY services/doctr/main.py .
COPY services/doctr/pyproject.toml .
COPY services/doctr/uv.lock .

ENTRYPOINT ["uv", "run", "python", "main.py"]
