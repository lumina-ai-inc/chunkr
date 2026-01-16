#!/bin/bash

# Script to run all workers locally in parallel for development

set -e

echo "🔨 Building all workers..."
cd "$(dirname "$0")/.."
cargo build --release --bin deal_document_worker --bin fact_extraction_worker

echo "🚀 Starting all workers in background..."
echo "📝 Make sure all services are running (Redis, Postgres, MinIO, Docling, AI services)"
echo "📝 Press Ctrl+C to stop all workers"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Function to cleanup background jobs on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all workers..."
    kill $(jobs -p) 2>/dev/null || true
    wait
    echo "✅ All workers stopped"
}

trap cleanup EXIT INT TERM

# Start workers in background
echo "📦 Starting deal document worker..."
./target/release/deal_document_worker &
DEAL_WORKER_PID=$!

echo "🤖 Starting fact extraction worker..."
./target/release/fact_extraction_worker &
FACT_WORKER_PID=$!

echo ""
echo "✅ All workers started:"
echo "   - Deal Document Worker (PID: $DEAL_WORKER_PID)"
echo "   - Fact Extraction Worker (PID: $FACT_WORKER_PID)"
echo ""
echo "📊 Monitoring logs (Ctrl+C to stop)..."

# Wait for all background jobs
wait
