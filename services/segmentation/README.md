# VGT Docker Setup

This guide explains how to set up and manage a VGT Docker container.


## Initial Setup

1. Create a history file:
```bash
touch ~/.bash_history_vgt
```

2. Create and run the doctr container:
```bash
sudo docker run -it --gpus all -p 8001:8000\
    --name vgt \
    -v $PWD:/app \
    -v ~/.bash_history_vgt:/root/.bash_history \
    pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime bash
```

### Starting the Services

1. **Inside Docker Container**
```bash
cd ../app

apt-get update && apt-get install -y -q --no-install-recommends \
    libgomp1 ffmpeg libsm6 libxext6 git ninja-build g++ || true
apt-get update --fix-missing

mkdir -p object_detection \
    && mkdir -p object_detection/weights
    && mkdir -p object_detection/configs

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install huggingface_hub==0.24.3
pip install wheel setuptools
pip install torch torchvision torchaudio
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@70f454304e1a38378200459dd2dbca0f0f4a5ab4'

pip --default-timeout=1000 install -r requirements.txt
python download_models.py

#entrypoint

python object_detection/server.py

# Container Management

### Starting the Container
```bash
sudo docker start -i vgt  
```

### Stopping and Cleaning Up
```bash
# Exit the container
exit

# Stop container (run this after exiting)
sudo docker stop vgt

# Optional: Remove the container and volume (run this after exiting)
sudo docker rm -v vgt
```
