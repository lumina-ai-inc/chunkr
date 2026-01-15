# Quick Start: Run Deal Document Worker

## The Problem
Documents were getting stuck at "⏳ Processing Document" because the worker wasn't running.

## The Solution
Run the deal document worker to process uploaded documents.

## Quick Start (Choose One)

### Option 1: Run Locally (Recommended for Development)

```bash
# Terminal 1: Start services (IMPORTANT - must be running first!)
make start

# Wait for services to start (30-60 seconds), then in Terminal 2:
./scripts/run-deal-worker-local.sh
```

**Important:** Make sure services are fully running before starting the worker. Check with:
```bash
docker compose ps
# All services should show "Up" status
```

### Option 2: Run Everything in Docker

```bash
# Build worker image first
docker build -f docker/deal-worker/Dockerfile -t luminainc/deal_document_worker:1.20.1 .

# Start all services (including worker)
make start

# View worker logs
docker compose logs -f deal-worker
```

## What You'll See

When a document is uploaded, the worker will print:
```
Received document processing message: {...}
Processing document doc-xxx for deal deal-yyy
Using OCR service: Azure Document Intelligence
Processing completed: 5 pages processed
Document processing completed: doc-xxx
```

## CSV Support Added ✨

The system now supports CSV files (like rent rolls):
- Upload a `.csv` file
- No OCR needed - instant processing
- Data extracted and ready for analysis

## Need Help?

See `OCR_PROCESSING_FIX.md` for detailed troubleshooting.

