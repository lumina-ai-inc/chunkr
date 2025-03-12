#!/bin/bash

# Save the current directory path
CURRENT_DIR=$(pwd)

cd ../../

# Define the Docker image name as a variable
DOCKER_IMAGE_NAME="luminainc/web"

# Get the current commit SHA if not already set
SHA=${GIT_SHA:-$(git rev-parse --short HEAD)}

# Get root version from manifest if not already set
if [ -z "$VERSION" ] && [ -f ".release-please-manifest.json" ]; then
    VERSION=$(grep -o '"\.": "[^"]*"' .release-please-manifest.json | cut -d'"' -f4)
fi

# Use VERSION or fall back to SHA
TAG=${VERSION:-$SHA}

echo "------------------------"
echo "Using tag: $TAG"
echo "------------------------"

# Build the Docker image with the tag
docker build --no-cache --platform linux/amd64 -t $DOCKER_IMAGE_NAME:$TAG -f $CURRENT_DIR/Dockerfile .

# Check if the build was successful
if [ $? -eq 0 ]; then
    # Push the Docker image with the tag
    docker push $DOCKER_IMAGE_NAME:$TAG
    
    # If we're building with a version, also tag with SHA for traceability
    if [ "$TAG" != "$SHA" ]; then
        docker tag $DOCKER_IMAGE_NAME:$TAG $DOCKER_IMAGE_NAME:$SHA
        docker push $DOCKER_IMAGE_NAME:$SHA
    fi

    # Also tag and push as latest
    docker tag $DOCKER_IMAGE_NAME:$TAG $DOCKER_IMAGE_NAME:latest
    docker push $DOCKER_IMAGE_NAME:latest
else
    echo "Docker build failed. Skipping push."
    exit 1
fi