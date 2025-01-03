FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y -q --no-install-recommends \
    libgomp1 ffmpeg libsm6 libxext6 git ninja-build g++ || true
RUN apt-get update --fix-missing

# Create all necessary directories first
RUN mkdir -p /app/src \
    && mkdir -p /app/models \
    && mkdir -p /app/object_detection \
    && mkdir -p /app/object_detection/weights

# Set up user and permissions
RUN addgroup --system python && adduser --system --group python
RUN chown -R python:python /app
USER python

ENV VIRTUAL_ENV=/app/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy environment variables
COPY .env .env

COPY requirements.txt requirements.txt

WORKDIR /app

# Copy the application code
COPY --chown=python:python . .
RUN pip install --upgrade pip
RUN pip install huggingface_hub==0.24.3

# Install build dependencies
RUN pip install wheel setuptools

# Install PyTorch first
RUN pip install torch torchvision torchaudio

# Install detectron2 from source with build dependencies
RUN . $VIRTUAL_ENV/bin/activate && \
    pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@70f454304e1a38378200459dd2dbca0f0f4a5ab4'

RUN pip --default-timeout=1000 install -r requirements.txt

# Download models with correct permissions
RUN python download_models.py

# Start the FastAPI server
ENTRYPOINT ["python", "object_detection/server.py"]