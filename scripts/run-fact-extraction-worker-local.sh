#!/bin/bash

# Script to run the fact extraction worker locally for development
# This worker processes fact extraction jobs from Redis queue

set -e

echo "🔨 Building fact extraction worker..."
cd "$(dirname "$0")/.."
cargo build --release --bin fact_extraction_worker

echo "🚀 Starting fact extraction worker..."
echo "📝 Make sure Redis, Postgres, and AI services (OpenAI/Anthropic) are configured"
echo "📝 Press Ctrl+C to stop"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run the worker
./target/release/fact_extraction_worker
