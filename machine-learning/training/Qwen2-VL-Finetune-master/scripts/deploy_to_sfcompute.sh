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

# Find the pod
echo "Finding the running pod..."
sleep 10
POD_NAME=$(kubectl get pods -n sf-lumina -l job-name=qwen-vl-training -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD_NAME" ]; then
  echo "Warning: Pod not found. You may need to manually set up port forwarding later."
  echo "Run: kubectl get pods -n sf-lumina"
  echo "Then: kubectl port-forward pod/POD_NAME 6006:6006 -n sf-lumina"
else
  echo "Pod found: $POD_NAME"
  echo "To set up TensorBoard port forwarding, run:"
  echo "kubectl port-forward pod/$POD_NAME 6006:6006 -n sf-lumina"
fi

echo "Deployment complete!" 