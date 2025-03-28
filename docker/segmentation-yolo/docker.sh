#!/bin/bash

# Save the current directory path
CURRENT_DIR=$(pwd)

cd ../../

# Define the Docker image name as a variable
DOCKER_IMAGE_NAME="luminainc/segmentation-yolo"

# Check if a custom tag was provided as a command line argument
CUSTOM_TAG=$1

# Get the current commit SHA if not already set
SHA=${GIT_SHA:-$(git rev-parse --short HEAD)}

# Get root version from manifest if not already set
if [ -z "$VERSION" ] && [ -f ".release-please-manifest.json" ]; then
    VERSION=$(grep -o '"\.": "[^"]*"' .release-please-manifest.json | cut -d'"' -f4)
fi

# Use CUSTOM_TAG first, then VERSION, then fall back to SHA
TAG=${CUSTOM_TAG:-${VERSION:-$SHA}}

echo "------------------------"
echo "Using tag: $TAG"
echo "------------------------"

# Build the Docker image with the TAG
docker build --platform linux/amd64 \
    --build-arg NVIDIA_VISIBLE_DEVICES=all \
    --build-arg NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -t $DOCKER_IMAGE_NAME:$TAG -f $CURRENT_DIR/Dockerfile .
    
# Check if the build was successful
if [ $? -eq 0 ]; then
    # Push the Docker image with the TAG
    docker push $DOCKER_IMAGE_NAME:$TAG

    # Optionally, you can also tag and push as latest
    docker tag $DOCKER_IMAGE_NAME:$TAG $DOCKER_IMAGE_NAME:latest
    docker push $DOCKER_IMAGE_NAME:latest
else
    echo "Docker build failed. Skipping push."
    exit 1
fi