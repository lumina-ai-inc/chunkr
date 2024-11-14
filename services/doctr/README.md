# Doctr Docker Setup

This guide explains how to set up and manage a Doctr Docker container..


## Initial Setup

1. Create a history file:
```bash
touch ~/.bash_history_doctr
```

2. Create and run the PaddleX container:
```bash
sudo docker run -it --gpus all -p 8000:8000\
    --name doctr \
    -v $PWD:/doctr \
    -v ~/.bash_history_doctr:/root/.bash_history \
    ghcr.io/mindee/doctr:tf-py3.8.18-gpu-2023-09 bash
```

### Starting the Services

1. **Inside Docker Container**
   ```bash
    # Install Rust and Cargo
    curl https://sh.rustup.rs -sSf | sh

    # Source the cargo environment
    source $HOME/.cargo/env

    # Navigate to the project directory
    cd /paddle

   # Start the PaddleX service proxy (choose ONE of the following):
   # For General OCR processing
   cargo r -- --pipeline ./config/OCR.yaml

   # For Table Recognition
   cargo r -- --pipeline ./config/table_recognition.yaml 
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