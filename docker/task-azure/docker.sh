#!/bin/bash

# Save the current directory path
CURRENT_DIR=$(pwd)

cd ../../

# Define the Docker image name as a variable
DOCKER_IMAGE_NAME="luminainc/task-azure"

# Get the current commit SHA if not already set
SHA=${GIT_SHA:-$(git rev-parse --short HEAD)}

# Get root version from manifest if not already set
if [ -z "$VERSION" ] && [ -f ".release-please-manifest-enterprise.json" ]; then
    VERSION=$(grep -o '"\.": "[^"]*"' .release-please-manifest-enterprise.json | cut -d'"' -f4)
fi

# Use VERSION or fall back to SHA
TAG=${VERSION:-$SHA}

echo "------------------------"
echo "Using tag: $TAG"
echo "------------------------"

# Build the Docker image with the tag
docker build --platform linux/amd64 --progress=plain -t $DOCKER_IMAGE_NAME:$TAG -f $CURRENT_DIR/Dockerfile .

# Check if the build was successful
if [ $? -eq 0 ]; then
    # tag as latest
    echo "------------------------"
    echo "Tagging $DOCKER_IMAGE_NAME:$TAG as $DOCKER_IMAGE_NAME:latest"
    echo "------------------------"
    docker tag $DOCKER_IMAGE_NAME:$TAG $DOCKER_IMAGE_NAME:latest
    
    # Push the Docker image with the tag    
    echo "------------------------"
    echo "Pushing image to $DOCKER_IMAGE_NAME:$TAG"
    echo "------------------------"
    docker push $DOCKER_IMAGE_NAME:$TAG

    echo "------------------------"
    echo "Pushing image to $DOCKER_IMAGE_NAME:latest"
    echo "------------------------"
    docker push $DOCKER_IMAGE_NAME:latest
else
    echo "Docker build failed. Skipping push."
    exit 1
fi