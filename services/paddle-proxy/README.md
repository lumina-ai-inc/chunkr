# PaddleX Docker Setup

This guide explains how to set up and manage a PaddleX Docker container and a proxy service.

## Initial Setup

1. Create a history file:
```bash
touch ~/.bash_history_paddlex
```

2. Create and run the PaddleX container:
```bash
sudo docker run --gpus all -p 8000:8000 \
    --name idk \
    -v $PWD:/paddle \
    -v ~/.bash_history_paddlex:/root/.bash_history \
    --shm-size=8g \
    --network=host \
    -it luminainc/general-ocr:latest \
    /bin/bash
```

## Development

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

### Verifying the Setup
1. The PaddleX service should be running on `http://localhost:8080`
2. The proxy service should be running on `http://localhost:8000`
3. You can test the connection by running:
   ```bash
   curl http://localhost:8000/health
   ```

## Testing

1. Install UV package manager:
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the cargo environment
source $HOME/.cargo/env
```

2. Run the following to test the proxy:

> **Note:** You need images in a ./input folder and the output will be in ./output

```bash
cd ./testing
python3 example.py
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