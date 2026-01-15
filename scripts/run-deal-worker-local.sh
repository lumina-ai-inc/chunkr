#!/bin/bash
# Script to run the deal document worker locally for testing

set -e

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "🔨 Building deal document worker..."
cd "$PROJECT_ROOT/core"
cargo build --release --bin deal_document_worker

echo ""
echo "🚀 Starting deal document worker..."
echo "📝 Make sure Redis, Postgres, and MinIO are running (via make start)"
echo "📝 Press Ctrl+C to stop"
echo ""

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
fi

# Run the worker from project root (needed for models.yaml)
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/core/target/release/deal_document_worker"

