FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    git \
    wget \
    python3.7 \
    python3.7-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Create and activate Conda environment
RUN conda create -n layoutreader python=3.7 -y
SHELL ["conda", "run", "-n", "layoutreader", "/bin/bash", "-c"]

# Install PyTorch with CUDA support
RUN conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch -y

# Install other dependencies
RUN pip install nltk transformers==2.10.0 packaging huggingface_hub

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Set working directory
WORKDIR /workspace

# Copy current directory contents into container
COPY . .

# Download model files directly
RUN mkdir -p /workspace/models/layoutreader-base-readingbank && cd /workspace/models/layoutreader-base-readingbank && \
    wget https://huggingface.co/LuminaInc/layoutreader-base-readingbank/resolve/main/config.json && \
    wget https://huggingface.co/LuminaInc/layoutreader-base-readingbank/resolve/main/pytorch_model.bin && \
    cd ../../

# Install Apex with better error handling and cleanup
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    git checkout 4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && \
    cd .. && \
    rm -rf apex

RUN pip install -e .

# Make conda environment active by default
RUN conda init bash && \
    echo "conda activate layoutreader" >> ~/.bashrc

# Ensure conda commands can be used in RUN instructions
SHELL ["conda", "run", "-n", "layoutreader", "/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]