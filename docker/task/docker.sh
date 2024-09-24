#!/bin/bash

CURRENT_DIR=$(pwd)

cd ../../

DOCKER_IMAGE_NAME="luminainc/task"

SHA=$(git rev-parse --short HEAD)
echo "------------------------"
echo $SHA
echo "------------------------"

docker build --platform linux/amd64 -t $DOCKER_IMAGE_NAME:$SHA -f $CURRENT_DIR/Dockerfile .

if [ $? -eq 0 ]; then
    docker push $DOCKER_IMAGE_NAME:$SHA

    docker tag $DOCKER_IMAGE_NAME:$SHA $DOCKER_IMAGE_NAME:latest
    docker push $DOCKER_IMAGE_NAME:latest
else
    echo "Docker build failed. Skipping push."
    exit 1
fi