.PHONY: help start start-no-build stop restart logs clean build-all build-backend build-frontend deploy status

# Detect if running on Mac (any architecture) and set compose files accordingly
# Mac doesn't have NVIDIA GPUs, so we use local override to remove GPU requirements
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  COMPOSE_FILES := -f docker-compose.production.yaml -f docker-compose.local.yaml
else
  COMPOSE_FILES := -f docker-compose.production.yaml
endif

# Default target
help:
	@echo "🚀 Orin Development & Deployment Commands"
	@echo "=========================================="
	@echo ""
	@echo "📝 Current Mode: $(if $(filter Darwin,$(UNAME_S)),LOCAL (Mac - GPU disabled),PRODUCTION (Linux - GPU enabled))"
	@echo "📝 Compose Files: $(COMPOSE_FILES)"
	@echo ""
	@echo "📦 Local Development (make start):"
	@echo "  - Base: docker-compose.production.yaml (all services defined)"
	@echo "  - Override: docker-compose.local.yaml (on Mac only - disables GPU services)"
	@echo "  - On Mac: GPU services set to replicas=0 (won't start)"
	@echo "  - On Linux: Full production config with GPU support"
	@echo "  - Access: http://localhost:5173 (Web), http://localhost:8000 (API)"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make start           - Start all services (LOCAL - auto-detects Mac/Linux)"
	@echo "  make stop            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make logs            - View logs from all services"
	@echo ""
	@echo "🔨 Building:"
	@echo "  make build-all       - Rebuild backend + frontend"
	@echo "  make build-backend   - Rebuild backend only"
	@echo "  make build-frontend  - Rebuild frontend only"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean           - Clean up containers and volumes"
	@echo "  make status          - Show status of all services"
	@echo ""
	@echo "☁️  Cloud Deployment:"
	@echo "  make deploy          - Deploy to GCP (uses gcp/cloudbuild.yaml + Terraform)"
	@echo "  Note: make start is for LOCAL development only, not cloud deployment"
	@echo ""

# Start all services (pulls/uses pre-built images)
start:
	@echo "🚀 Starting Orin services (LOCAL MODE)..."
	@echo "📝 Using: $(COMPOSE_FILES)"
	@echo "🧹 Stopping any existing containers..."
	@docker compose $(COMPOSE_FILES) down --remove-orphans 2>/dev/null || true
	@echo "🚀 Starting services..."
	@docker compose $(COMPOSE_FILES) up -d --force-recreate --remove-orphans
	@echo ""
	@echo "✅ Services started!"
	@echo ""
	@echo "📱 Access points:"
	@echo "   Web UI:   http://localhost:5173"
	@echo "   API:      http://localhost:8000"
	@echo "   Keycloak: http://localhost:8080"
	@echo "   MinIO:    http://localhost:9001"
	@echo "   Qdrant:   http://localhost:6333"
	@echo ""
	@echo "💡 Tip: Use 'make logs' to view service logs"

# Quick start without rebuilding (fastest for daily development)
start-no-build:
	@echo "⚡ Quick starting Orin (using existing images)..."
	@docker compose $(COMPOSE_FILES) up -d --no-build
	@echo "✅ Services started!"

# Stop all services
stop:
	@echo "🛑 Stopping Orin services..."
	@docker compose $(COMPOSE_FILES) down --remove-orphans
	@echo "🧹 Removing any containers still using ports..."
	@docker ps -a --filter "name=data-extract" --format "{{.Names}}" | xargs -r docker rm -f 2>/dev/null || true
	@echo "✅ All services stopped"

# Restart all services
restart: stop start

# View logs
logs:
	@echo "📋 Viewing logs (Ctrl+C to exit)..."
	@docker compose $(COMPOSE_FILES) logs -f

# Show status of all services
status:
	@echo "📊 Service Status:"
	@echo ""
	@docker compose $(COMPOSE_FILES) ps
	@echo ""
	@echo "🐳 Docker Containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" --filter "name=orin" 2>/dev/null || docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Build all (backend + frontend)
build-all: build-backend build-frontend
	@echo "✅ All builds complete!"

# Build backend
build-backend:
	@echo "🔨 Building backend..."
	@cd core && cargo build --release
	@echo "✅ Backend build complete!"

# Build frontend
build-frontend:
	@echo "🔨 Building frontend..."
	@cd apps/web && npm install --legacy-peer-deps && npm run build
	@echo "✅ Frontend build complete!"

# Clean up everything
clean:
	@echo "🧹 Cleaning up Docker resources..."
	@docker compose $(COMPOSE_FILES) down -v --remove-orphans 2>/dev/null || true
	@echo "🧹 Removing any orphaned containers..."
	@docker ps -a --filter "name=data-extract" --format "{{.ID}}" | xargs docker rm -f 2>/dev/null || true
	@echo "🧹 Removing dangling images..."
	@docker image prune -f
	@echo "✅ Cleanup complete!"

# Deploy to GCP
deploy:
	@echo "☁️  Deploying to GCP..."
	@./scripts/deploy-gcp.sh

