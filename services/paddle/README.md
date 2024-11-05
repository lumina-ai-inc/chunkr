# PaddleX Docker Setup

This guide explains how to set up and manage a PaddleX Docker container for table OCR processing.

## Initial Setup

1. Create a history file:
```bash
touch ~/.bash_history_paddlex
```

2. Create and run the PaddleX container:
```bash
sudo docker run --gpus all -p 8000:8000 \
    --name paddlex \
    -v $PWD:/paddle \
    -v ~/.bash_history_paddlex:/root/.bash_history \
    --shm-size=8g \
    --network=host \
    -it registry.baidubce.com/paddlex/paddlex:paddlex3.0.0b1-paddlepaddle3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6 \
    /bin/bash
```

## Container Management

### Starting the Container
```bash
sudo docker start -i paddlex
```

### Stopping and Cleaning Up
```bash
# Exit the container
exit

# Stop container (run this after exiting)
sudo docker stop paddlex

# Optional: Remove the container and volume (run this after exiting)
sudo docker rm -v paddlex
```

## Development

1. Install UV package manager:
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the cargo environment
source $HOME/.cargo/env
```

2. Navigate to the project directory:
```bash
cd /paddle
```

3. Run services:
```bash
cd /paddle

# Choose one of the following:
## General OCR
paddlex --pipeline ./config/OCR.yaml --serve --port 8000

## Table Recognition
paddlex --pipeline ./config/table_recognition.yaml --serve --port 8000

## Run local implementation
uv run main.py
```

Alternatively, you can run the local implementation without using UV:
```bash
cd /paddle

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python3 main.py
```
