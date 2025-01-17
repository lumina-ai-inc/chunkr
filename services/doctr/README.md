# Doctr Docker Setup

This guide explains how to set up and manage a Doctr Docker container.


## Initial Setup

1. Create a history file:
```bash
touch ~/.bash_history_doctr
```

2. Create and run the doctr container:
```bash
sudo docker run -it --gpus all -p 8002:8000\
    --name doctr \
    -v $PWD:/doctr \
    -v ~/.bash_history_doctr:/root/.bash_history \
    ghcr.io/mindee/doctr:torch-py3.9.18-gpu-2023-09 bash
```

### Starting the Services

1. **Inside Docker Container**
```bash
# Install Rust and Cargo
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the cargo environment
source $HOME/.local/bin/env

cd /doctr

# For General OCR processing
uv run python main.py
```

# Container Management

### Starting the Container
```bash
sudo docker start -i doctr  
```

### Stopping and Cleaning Up
```bash
# Exit the container
exit

# Stop container (run this after exiting)
sudo docker stop doctr

# Optional: Remove the container and volume (run this after exiting)
sudo docker rm -v doctr
```