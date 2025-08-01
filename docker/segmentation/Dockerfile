FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y -q --no-install-recommends \
    libgomp1 ffmpeg libsm6 libxext6 git ninja-build g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/yolo/download_model.py ./download_model.py
COPY services/yolo/main.py ./main.py
COPY services/yolo/pyproject.toml ./pyproject.toml
COPY services/yolo/models.py ./models.py
COPY services/yolo/prediction.py ./prediction.py

RUN pip install --upgrade pip && \
    pip install wheel setuptools && \
    pip install uv && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install python-dotenv opencv-python-headless fastapi uvicorn pydantic huggingface_hub numpy python-multipart scikit-learn && \
    pip install ultralytics && \
    pip install doclayout-yolo>=0.0.3

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

RUN uv run download_model.py

ENTRYPOINT ["uv", "run", "main.py"]