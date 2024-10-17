# Use CUDA-enabled PyTorch image as the base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the model
RUN wget -P /app/models/table-structure-recognition-v1.1-all https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all/resolve/main/config.json
RUN wget -P /app/models/table-structure-recognition-v1.1-all https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all/resolve/main/model.safetensors

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]