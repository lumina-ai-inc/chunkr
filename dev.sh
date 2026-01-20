#!/bin/bash
set -e

echo "🚀 ReFlow Development Environment Setup"
echo "========================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "   Please start Docker Desktop and try again"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "   Copying from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "   ✅ Created .env - please edit with your API keys"
    else
        echo "   ❌ .env.example not found - you'll need to create .env manually"
    fi
    echo ""
fi

# Check if first time setup is needed
if [ ! -d "core/target/release" ] || [ ! -d "apps/web/dist" ]; then
    echo "📦 First time setup detected"
    echo "   This will take 5-10 minutes..."
    echo ""
    read -p "   Run 'make build-all' now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        make build-all
        echo ""
        echo "✅ Build complete!"
        echo ""
    else
        echo "⚠️  Skipping build - you'll need to run 'make build-all' manually"
        echo ""
    fi
fi

# Start Docker services
echo "📦 Starting Docker services..."
echo "   - PostgreSQL (database)"
echo "   - Redis (queue)"
echo "   - MinIO (storage)"
echo "   - Qdrant (vector DB)"
echo "   - Docling (OCR service)"
echo ""

docker compose -f docker-compose.local.yaml up -d postgres redis minio qdrant docling-service keycloak

# Wait for services with health checks
echo "⏳ Waiting for services to be ready..."
sleep 3

# Check Postgres
echo -n "   Checking PostgreSQL... "
for i in {1..10}; do
    if docker compose -f docker-compose.local.yaml exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo "✅"
        break
    fi
    sleep 1
done

# Check Redis
echo -n "   Checking Redis... "
if docker compose -f docker-compose.local.yaml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅"
fi

# Check Docling
echo -n "   Checking Docling OCR... "
for i in {1..15}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "✅"
        break
    fi
    sleep 1
done

echo ""
echo "✅ All services started successfully!"
echo ""
echo "========================================"
echo "📝 Next Steps - Open 3 Terminals:"
echo "========================================"
echo ""
echo "Terminal 1 (Backend API):"
echo "  cd core && cargo run"
echo ""
echo "Terminal 2 (Workers):"
echo "  ./scripts/run-all-workers-local.sh"
echo ""
echo "Terminal 3 (Frontend):"
echo "  cd apps/web && npm run dev"
echo ""
echo "========================================"
echo "🌐 Application URLs:"
echo "========================================"
echo ""
echo "  Frontend:    http://localhost:5173"
echo "  Backend API: http://localhost:8000"
echo "  API Docs:    http://localhost:8000/docs"
echo "  Docling OCR: http://localhost:8002"
echo ""
echo "========================================"
echo "🛠️  Useful Commands:"
echo "========================================"
echo ""
echo "  Check services:  docker compose -f docker-compose.local.yaml ps"
echo "  View logs:       docker compose -f docker-compose.local.yaml logs -f"
echo "  Stop services:   docker compose -f docker-compose.local.yaml down"
echo "  Clean restart:   docker compose -f docker-compose.local.yaml down -v && ./dev.sh"
echo ""
echo "📖 For more info, see DEVELOPMENT_GUIDE.md"
echo ""
