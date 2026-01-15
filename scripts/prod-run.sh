#!/bin/bash

# Production deployment script with comprehensive options
# Usage: ./scripts/prod-run.sh [option]
# Options: deploy, rebuild-all, stop-all, restart-all, status, logs, clean

set -e

# Configuration
NAMESPACE="chunkr"
RELEASE_NAME="chunkr"
CHART_PATH="./kube/charts/chunkr"
VALUES_FILE="./kube/charts/chunkr/values.yaml"
INFRASTRUCTURE_FILE="./kube/charts/chunkr/infrastructure.yaml"
CURRENT_VALUES_FILE="./current-values.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Get the command option
OPTION=${1:-deploy}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        print_error "Helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        echo "Please ensure kubectl is configured for your GKE cluster"
        exit 1
    fi
    
    CURRENT_CONTEXT=$(kubectl config current-context)
    print_success "Connected to cluster: $CURRENT_CONTEXT"
    
    # Check Docker registry login
    print_info "Checking Docker registry access..."
    if ! docker info | grep -q "Username"; then
        print_error "Not logged into Docker registry"
        echo "Please run: docker login ghcr.io"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to get version
get_version() {
    if [ -f ".release-please-manifest.json" ]; then
        VERSION=$(grep -o '"\.": "[^"]*"' .release-please-manifest.json | cut -d'"' -f4)
    else
        VERSION=$(date +%Y%m%d-%H%M%S)
    fi
    echo $VERSION
}

# Function to rebuild all images
rebuild_all() {
    print_info "Rebuilding all production images..."
    
    VERSION=$(get_version)
    GIT_SHA=$(git rev-parse --short HEAD)
    
    print_info "Version: $VERSION"
    print_info "Git SHA: $GIT_SHA"
    print_info "Registry: ghcr.io/buildorin"
    echo ""
    
    check_prerequisites
    
    # Function to build and push image
    build_and_push() {
        local service=$1
        local dockerfile_path=$2
        local image_name="ghcr.io/buildorin/orin-${service}"
        
        print_info "Building ${service} image..."
        echo "Image: ${image_name}:${VERSION}"
        
        # Build for linux/amd64 (GKE production)
        docker build \
            --platform linux/amd64 \
            --no-cache \
            -t "${image_name}:${VERSION}" \
            -t "${image_name}:latest" \
            -t "${image_name}:${GIT_SHA}" \
            -f "${dockerfile_path}" \
            .
        
        if [ $? -eq 0 ]; then
            print_success "Build successful for ${service}"
            
            # Push all tags
            print_info "Pushing ${service} images..."
            docker push "${image_name}:${VERSION}"
            docker push "${image_name}:latest"
            docker push "${image_name}:${GIT_SHA}"
            
            print_success "Successfully pushed ${service} images"
        else
            print_error "Build failed for ${service}"
            exit 1
        fi
        
        echo ""
    }
    
    # Build webapp image
    print_info "Building webapp image..."
    build_and_push "web" "docker/web/Dockerfile"
    
    # Build server image
    print_info "Building server image..."
    build_and_push "server" "docker/server/Dockerfile"
    
    # Build task image
    print_info "Building task image..."
    build_and_push "task" "docker/task/Dockerfile"
    
    print_success "All images rebuilt and pushed successfully"
}

# Function to stop all services
stop_all() {
    print_info "Stopping all production services..."
    
    check_prerequisites
    
    CURRENT_CONTEXT=$(kubectl config current-context)
    print_warning "This will stop all services in cluster: $CURRENT_CONTEXT"
    read -p "Are you sure? (y/N): " CONFIRM
    if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
        print_info "Stop cancelled"
        exit 0
    fi
    
    # Scale down all deployments
    print_info "Scaling down deployments..."
    kubectl scale deployment --all --replicas=0 -n $NAMESPACE 2>/dev/null || true
    
    print_success "All services stopped"
}

# Function to deploy
deploy() {
    print_info "Production Deployment Pipeline"
    echo "================================="
    
    VERSION=$(get_version)
    print_info "Version: $VERSION"
    print_info "Registry: ghcr.io/buildorin"
    print_info "Namespace: $NAMESPACE"
    echo ""
    
    check_prerequisites
    
    CURRENT_CONTEXT=$(kubectl config current-context)
    print_warning "Deploy to production cluster '$CURRENT_CONTEXT'?"
    read -p "Continue? (y/N): " CONFIRM
    if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
        print_info "Deployment cancelled"
        exit 0
    fi
    
    echo ""
    print_info "Step 1: Building production images..."
    echo "======================================="
    rebuild_all
    
    echo ""
    print_info "Step 2: Deploying to GKE cluster..."
    echo "======================================"
    
    # Check if namespace exists
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        print_info "Creating namespace $NAMESPACE..."
        kubectl create namespace $NAMESPACE
    else
        print_success "Namespace $NAMESPACE already exists"
    fi
    
    # Check if secrets exist
    print_info "Checking secrets..."
    if ! kubectl get secret orin-secret -n $NAMESPACE &> /dev/null; then
        print_error "Secret 'orin-secret' not found in namespace $NAMESPACE"
        echo "Please create the secret first. Check kube/README.md for instructions."
        exit 1
    fi
    
    # Check if LLM models configmap exists
    if ! kubectl get configmap llm-models-configmap -n $NAMESPACE &> /dev/null; then
        print_error "ConfigMap 'llm-models-configmap' not found in namespace $NAMESPACE"
        echo "Please create the configmap first. Check kube/README.md for instructions."
        exit 1
    fi
    
    print_success "Required secrets and configmaps found"
    
    # Deploy to cluster
    print_info "Deploying with images:"
    echo "  Web: ghcr.io/buildorin/orin-web:$VERSION"
    echo "  Server: ghcr.io/buildorin/orin-server:$VERSION"
    echo "  Task: ghcr.io/buildorin/orin-task:$VERSION"
    
    helm upgrade --install $RELEASE_NAME $CHART_PATH \
        --namespace $NAMESPACE \
        --values $VALUES_FILE \
        --values $INFRASTRUCTURE_FILE \
        --values $CURRENT_VALUES_FILE \
        --set "services.web.image.tag=$VERSION" \
        --set "services.server.image.tag=$VERSION" \
        --set "services.task.image.tag=$VERSION" \
        --wait \
        --timeout 10m
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Production deployment completed successfully!"
        echo ""
        show_status
    else
        print_error "Deployment failed"
        exit 1
    fi
}

# Function to restart all services
restart_all() {
    print_info "Restarting all production services..."
    
    check_prerequisites
    
    CURRENT_CONTEXT=$(kubectl config current-context)
    print_warning "This will restart all services in cluster: $CURRENT_CONTEXT"
    read -p "Continue? (y/N): " CONFIRM
    if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
        print_info "Restart cancelled"
        exit 0
    fi
    
    # Restart all deployments
    print_info "Restarting deployments..."
    kubectl rollout restart deployment -n $NAMESPACE
    
    print_info "Waiting for rollouts to complete..."
    kubectl rollout status deployment -n $NAMESPACE --timeout=10m
    
    print_success "All services restarted"
}

# Function to show status
show_status() {
    print_info "Production Service Status:"
    echo ""
    
    check_prerequisites
    
    echo "📊 Pods:"
    kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME
    
    echo ""
    echo "🚀 Deployments:"
    kubectl get deployments -n $NAMESPACE
    
    echo ""
    echo "🔗 Services:"
    kubectl get services -n $NAMESPACE
    
    echo ""
    echo "🌐 Ingress:"
    kubectl get ingress -n $NAMESPACE
    
    echo ""
    echo "📋 Management Commands:"
    echo "  View all resources: kubectl get all -n $NAMESPACE"
    echo "  View server logs:   kubectl logs -f deployment/$RELEASE_NAME-server -n $NAMESPACE"
    echo "  View web logs:      kubectl logs -f deployment/$RELEASE_NAME-web -n $NAMESPACE"
    echo "  View task logs:     kubectl logs -f deployment/$RELEASE_NAME-task -n $NAMESPACE"
    echo "  Scale deployment:   kubectl scale deployment/$RELEASE_NAME-server --replicas=3 -n $NAMESPACE"
}

# Function to show logs
show_logs() {
    SERVICE=${2:-server}
    
    check_prerequisites
    
    case $SERVICE in
        server|backend)
            print_info "Showing logs for server..."
            kubectl logs -f deployment/$RELEASE_NAME-server -n $NAMESPACE
            ;;
        web|frontend)
            print_info "Showing logs for web..."
            kubectl logs -f deployment/$RELEASE_NAME-web -n $NAMESPACE
            ;;
        task)
            print_info "Showing logs for task..."
            kubectl logs -f deployment/$RELEASE_NAME-task -n $NAMESPACE
            ;;
        all)
            print_info "Showing logs for all services (Ctrl+C to exit)..."
            kubectl logs -f deployment/$RELEASE_NAME-server -n $NAMESPACE &
            kubectl logs -f deployment/$RELEASE_NAME-web -n $NAMESPACE &
            kubectl logs -f deployment/$RELEASE_NAME-task -n $NAMESPACE &
            wait
            ;;
        *)
            print_error "Unknown service: $SERVICE"
            echo "Available services: server, backend, web, frontend, task, all"
            exit 1
            ;;
    esac
}

# Function to clean everything
clean_all() {
    print_warning "This will DELETE all production resources in namespace: $NAMESPACE"
    check_prerequisites
    
    CURRENT_CONTEXT=$(kubectl config current-context)
    print_error "WARNING: This will delete everything in cluster: $CURRENT_CONTEXT"
    read -p "Type 'DELETE' to confirm: " CONFIRM
    if [ "$CONFIRM" != "DELETE" ]; then
        print_info "Clean cancelled"
        exit 0
    fi
    
    print_info "Uninstalling Helm release..."
    helm uninstall $RELEASE_NAME -n $NAMESPACE 2>/dev/null || true
    
    print_info "Deleting namespace..."
    kubectl delete namespace $NAMESPACE 2>/dev/null || true
    
    print_success "Clean completed"
}

# Function to show help
show_help() {
    echo "Production Run Script"
    echo "===================="
    echo ""
    echo "Usage: ./scripts/prod-run.sh [option]"
    echo ""
    echo "Options:"
    echo "  deploy         Build images and deploy to production (default)"
    echo "  rebuild-all    Rebuild and push all production images"
    echo "  stop-all       Stop all running services (scale to 0)"
    echo "  restart-all    Restart all services (rollout restart)"
    echo "  status         Show status of all services"
    echo "  logs [service] Show logs for a service (server, web, task, all)"
    echo "  clean          Delete all production resources (DANGEROUS)"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./scripts/prod-run.sh deploy"
    echo "  ./scripts/prod-run.sh rebuild-all"
    echo "  ./scripts/prod-run.sh logs server"
    echo "  ./scripts/prod-run.sh status"
}

# Main command handler
case $OPTION in
    deploy)
        deploy
        ;;
    rebuild-all|rebuild)
        rebuild_all
        ;;
    stop-all|stop)
        stop_all
        ;;
    restart-all|restart)
        restart_all
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$@"
        ;;
    clean)
        clean_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown option: $OPTION"
        echo ""
        show_help
        exit 1
        ;;
esac
