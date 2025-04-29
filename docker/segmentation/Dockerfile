FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y -q --no-install-recommends \
    libgomp1 ffmpeg libsm6 libxext6 git ninja-build g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY services/vgt/object_detection/configs ./object_detection/configs
COPY services/vgt/object_detection/ditod ./object_detection/ditod
COPY services/vgt/object_detection/memory_inference.py ./object_detection/memory_inference.py
COPY services/vgt/object_detection/server.py ./object_detection/server.py
COPY services/vgt/object_detection/tokenization.py ./object_detection/tokenization.py
COPY services/vgt/configuration.py configuration.py
COPY services/vgt/download_models.py download_models.py
COPY services/vgt/requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install huggingface_hub==0.24.3 wheel setuptools && \
    pip install torch torchvision torchaudio && \
    pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@70f454304e1a38378200459dd2dbca0f0f4a5ab4' && \
    pip --default-timeout=1000 install -r requirements.txt && \
    python download_models.py

ENTRYPOINT ["python", "object_detection/server.py"]