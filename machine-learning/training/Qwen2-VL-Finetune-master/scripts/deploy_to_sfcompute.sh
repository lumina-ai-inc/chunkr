#!/bin/bash

# Load environment variables
set -a
source .env
set +a

# Configuration
DOCKER_REGISTRY="luminainc"
IMAGE_NAME="qwen-vl-finetune"
TAG="latest"
FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${IMAGE_NAME}:${TAG}"

# Optional Docker build and push
# Ask user if they want to build and push the Docker image
read -p "Do you want to build and push the Docker image? (y/n): " BUILD_RESPONSE
if [[ "$BUILD_RESPONSE" =~ ^[Yy]$ ]]; then
  # Login to Docker registry
  echo "Logging in to Docker registry..."
  sudo docker login -u ${LUMINA_DOCKERHUB_USERNAME} -p ${LUMINA_DOCKERHUB_PASSWORD}

  # Build the Docker image
  echo "Building Docker image..."
  sudo docker buildx build --platform linux/amd64 -t ${FULL_IMAGE_NAME} .

  # Push the image to registry
  echo "Pushing image to registry..."
  sudo docker push ${FULL_IMAGE_NAME}
fi

# Process and apply Kubernetes secrets
echo "Processing and applying secrets..."
envsubst < kube/secrets.yaml > kube/secrets_processed.yaml
envsubst < kube/luminadockerhub.yaml > kube/luminadockerhub_processed.yaml
kubectl apply -f kube/secrets_processed.yaml
kubectl apply -f kube/luminadockerhub_processed.yaml

# Delete existing job if it exists (Jobs are immutable in Kubernetes)
echo "Checking for existing job..."
if kubectl get job qwen-vl-training -n sf-lumina &> /dev/null; then
  echo "Deleting existing job..."
  kubectl delete job qwen-vl-training -n sf-lumina
  # Wait for job to be fully deleted
  kubectl wait --for=delete job/qwen-vl-training -n sf-lumina --timeout=60s
fi

# Apply the Kubernetes manifest
echo "Deploying to SF Compute..."
kubectl apply -f kube/training.yaml

# Clean up
rm kube/secrets_processed.yaml kube/luminadockerhub_processed.yaml

# Setup port forwarding for TensorBoard
echo "Ensure you are in the right context"
echo "finding pods"
kubectl get pods


echo "Setting up port forwarding for TensorBoard..."
echo "To access TensorBoard, run: kubectl port-forward pod/NAMEOFPOD 6006:6006"

echo "Deployment complete!" 