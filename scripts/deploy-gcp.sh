#!/bin/bash
set -e

# Orin GCP Deployment Script
# This script builds, pushes, and deploys the Orin application to GCP

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌  $1${NC}"
}

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed. Please install it first."
    print_info "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get GCP project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    print_error "No GCP project is set. Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

print_info "Deploying to GCP project: $PROJECT_ID"

# Confirm deployment
read -p "Are you sure you want to deploy to production? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    print_warning "Deployment cancelled"
    exit 0
fi

# Step 1: Build containers using Cloud Build
print_info "Building containers with Cloud Build..."
gcloud builds submit --config=gcp/cloudbuild.yaml .

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

print_success "Containers built and pushed to GCR"

# Step 2: Update Compute Engine instance
print_info "Updating Compute Engine instance..."
gcloud compute instances update-container orin-production \
    --zone=us-central1-a \
    --container-manifest=gcp/terraform/container-manifest.yaml

if [ $? -ne 0 ]; then
    print_error "Failed to update instance"
    exit 1
fi

print_success "Instance updated"

# Step 3: Run database migrations
print_info "Running database migrations..."
# SSH into the instance and run migrations
gcloud compute ssh orin-production \
    --zone=us-central1-a \
    --command="docker exec \$(docker ps -qf name=orin-server) /app/server migrate"

if [ $? -ne 0 ]; then
    print_warning "Migration command failed or not applicable"
fi

# Step 4: Health check
print_info "Performing health check..."
INSTANCE_IP=$(gcloud compute instances describe orin-production \
    --zone=us-central1-a \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

sleep 10  # Wait for services to start

for i in {1..5}; do
    if curl -sf "http://$INSTANCE_IP:8000/health" > /dev/null; then
        print_success "Health check passed!"
        break
    else
        print_warning "Health check attempt $i/5 failed, retrying..."
        sleep 5
    fi
    
    if [ $i -eq 5 ]; then
        print_error "Health check failed after 5 attempts"
        print_info "Please check logs: gcloud compute ssh orin-production --zone=us-central1-a"
        exit 1
    fi
done

# Print deployment info
print_success "Deployment complete!"
echo ""
echo "📱 Access your deployment at:"
echo "   Web UI:   http://$INSTANCE_IP:5173"
echo "   API:      http://$INSTANCE_IP:8000"
echo "   Keycloak: http://$INSTANCE_IP:8080"
echo ""
echo "📋 Useful commands:"
echo "   View logs:      gcloud compute ssh orin-production --zone=us-central1-a"
echo "   View instance:  gcloud compute instances describe orin-production --zone=us-central1-a"
echo "   Stop instance:  gcloud compute instances stop orin-production --zone=us-central1-a"
echo ""

