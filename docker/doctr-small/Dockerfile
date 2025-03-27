FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY services/doctr/.python-version .
COPY services/doctr/download_models.py .
COPY services/doctr/main.py .
COPY services/doctr/pyproject.toml .

ENV OCR_DETECTION_MODEL=fast_base
ENV OCR_RECOGNITION_MODEL=vitstr_small

RUN uv run download_models.py

EXPOSE 8000

ENTRYPOINT ["uv", "run", "python3", "main.py"]