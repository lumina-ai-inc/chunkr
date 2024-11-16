FROM ghcr.io/mindee/doctr:torch-py3.9.18-gpu-2023-09

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY services/doctr/.python-version .
COPY services/doctr/download_models.py .
COPY services/doctr/main.py .
COPY services/doctr/pyproject.toml .
COPY services/doctr/uv.lock .

RUN uv run download_models.py

EXPOSE 8000

ENTRYPOINT ["uv", "run", "python3", "main.py"]