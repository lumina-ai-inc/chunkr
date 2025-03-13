FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y -q --no-install-recommends \
    libgomp1 ffmpeg libsm6 libxext6 git ninja-build g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/yolo/main.py ./main.py
COPY services/yolo/pyproject.toml ./pyproject.toml

RUN pip install --upgrade pip && \
    pip install wheel setuptools && \
    pip install uv && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install opencv-python-headless fastapi uvicorn pydantic huggingface_hub numpy python-multipart && \
    pip install ultralytics && \
    pip install doclayout-yolo>=0.0.3

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["uv", "run", "main.py"]