# Testing Document Processing Flow

## Overview

This guide will help you test the complete document processing pipeline from upload to fact extraction.

## Prerequisites

### 1. Services Running

Make sure all required services are running:

```bash
# Terminal 1: Core services (Postgres, Redis, MinIO, Qdrant)
make start

# Terminal 2: Docling OCR service
cd services/docling-service
pip install -r requirements.txt
python app.py
# Should start on http://localhost:8002

# Terminal 3: Backend API server
cd core
cargo run
# Should start on http://localhost:8000

# Terminal 4: Workers (both deal document and fact extraction)
./scripts/run-all-workers-local.sh

# Terminal 5: Frontend
cd apps/web
npm run dev
# Should start on http://localhost:5173
```

### 2. Environment Variables

Ensure your `.env` file has:

```bash
# OCR Service
OCR_PROVIDER=docling
DOCLING_SERVICE_URL=http://localhost:8002

# AI Services (for fact extraction)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres

# Redis
REDIS__URL=redis://localhost:6379
```

## Testing Steps

### Test 1: Verify Services Health

```bash
# Check Docling service
curl http://localhost:8002/health
# Expected: {"status": "healthy", "gpu_available": false}

# Check backend API
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Check Redis
redis-cli ping
# Expected: PONG

# Check Postgres
psql postgresql://postgres:postgres@localhost:5432/postgres -c "SELECT 1;"
# Expected: 1
```

### Test 2: Upload Document via Frontend

1. Open browser: `http://localhost:5173`
2. Login (if auth is enabled)
3. Navigate to Dashboard
4. Click "+ New Deal"
5. Enter deal name: "Test Property"
6. Click "Create"
7. Upload a PDF document (rent roll, P&L, or any real estate document)
8. Wait for processing

### Test 3: Monitor Processing

Watch the worker logs in Terminal 4:

```
Expected output sequence:

[Deal Document Worker]
Received document processing message: {"document_id":"...","deal_id":"..."}
Downloaded document from S3: ...
[METRIC] {"metric":"s3_download_duration_ms",...}
Using Docling service: Docling Document Understanding
[METRIC] {"metric":"ocr_duration_ms",...}
OCR results stored for document ...
Fact extraction triggered for document ...
[METRIC] {"metric":"pipeline_complete",...}

[Fact Extraction Worker]
Received fact extraction message: {"document_id":"...","deal_id":"..."}
Processing fact extraction for document ...
Fact extraction completed: X facts extracted
```

### Test 4: Verify in Database

```bash
# Check documents table
psql postgresql://postgres:postgres@localhost:5432/postgres -c \
  "SELECT document_id, file_name, status, page_count FROM documents ORDER BY created_at DESC LIMIT 5;"

# Check facts table
psql postgresql://postgres:postgres@localhost:5432/postgres -c \
  "SELECT fact_id, label, value, confidence_score FROM facts ORDER BY created_at DESC LIMIT 10;"
```

### Test 5: Verify in Frontend

1. Navigate to the deal in Dashboard
2. Click on "Facts" tab
3. You should see extracted facts with:
   - Label (e.g., "Gross Rent", "Operating Expenses")
   - Value (e.g., "$378,000")
   - Confidence score
   - Source citation (document name, page number)
4. Click "Approve" on facts
5. Navigate to "Analysis" tab
6. Click "Run Underwriting"
7. Verify financial metrics are calculated

## Expected Flow

```
User Upload
    ↓
POST /api/v1/deals/:id/documents
    ↓
Store in MinIO
    ↓
Queue: deal_documents (Redis)
    ↓
Deal Document Worker
    ↓
Download from MinIO
    ↓
Docling OCR Service (http://localhost:8002/convert)
    ↓
Store OCR results in DB
    ↓
Queue: fact_extraction (Redis)
    ↓
Fact Extraction Worker
    ↓
AI Agent (OpenAI GPT-4)
    ↓
Store facts in DB
    ↓
Update document status: completed
    ↓
Frontend displays facts
```

## Troubleshooting

### Issue: "Processing Failed" in Frontend

**Check:**
1. Worker logs for errors
2. Docling service is running: `curl http://localhost:8002/health`
3. Redis queue has messages: `redis-cli LLEN deal_documents`

### Issue: No facts extracted

**Check:**
1. Fact extraction worker is running
2. OpenAI API key is valid: `echo $OPENAI_API_KEY`
3. Check worker logs for AI errors

### Issue: "Failed to download from S3"

**Check:**
1. MinIO is running: `curl http://localhost:9000/minio/health/live`
2. Environment variables are correct
3. File was uploaded successfully

### Issue: Worker crashes on startup

**Check:**
1. Rust compilation succeeded: `cargo build --release --bin deal_document_worker`
2. Database migrations ran: `cd core && diesel migration run`
3. All dependencies are installed

## Performance Metrics

Monitor these metrics in worker logs:

- `s3_download_duration_ms`: Should be < 1000ms for typical documents
- `ocr_duration_ms`: Should be < 10000ms (10s) for typical PDFs
- `pipeline_complete.total_duration_ms`: Should be < 30000ms (30s) total

## Success Criteria

✅ **Complete Success When:**
1. Document uploads without errors
2. Worker processes document (check logs)
3. OCR results stored in database
4. Facts extracted and visible in UI
5. Facts can be approved
6. Underwriting can be run

## Next Steps

After successful testing:
1. Test with different document types (CSV, Excel, multi-page PDFs)
2. Test error handling (upload invalid file, stop Docling service mid-processing)
3. Test with multiple concurrent uploads
4. Monitor resource usage (CPU, memory, Redis queue size)

## Debugging Commands

```bash
# Check Redis queues
redis-cli LLEN deal_documents
redis-cli LLEN fact_extraction

# View queue contents (non-destructive)
redis-cli LRANGE deal_documents 0 -1
redis-cli LRANGE fact_extraction 0 -1

# Check recent documents
psql postgresql://postgres:postgres@localhost:5432/postgres -c \
  "SELECT document_id, file_name, status, created_at FROM documents ORDER BY created_at DESC LIMIT 5;"

# Check recent facts
psql postgresql://postgres:postgres@localhost:5432/postgres -c \
  "SELECT fact_id, label, value, status, created_at FROM facts ORDER BY created_at DESC LIMIT 10;"

# View worker logs
tail -f /path/to/worker/logs

# Check Docling service logs
# (In Terminal 2 where Docling is running)
```

## Clean Up

To reset for fresh testing:

```bash
# Clear Redis queues
redis-cli DEL deal_documents
redis-cli DEL fact_extraction

# Delete test documents and facts
psql postgresql://postgres:postgres@localhost:5432/postgres -c \
  "DELETE FROM facts WHERE deal_id IN (SELECT deal_id FROM deals WHERE deal_name LIKE 'Test%');"
psql postgresql://postgres:postgres@localhost:5432/postgres -c \
  "DELETE FROM documents WHERE deal_id IN (SELECT deal_id FROM deals WHERE deal_name LIKE 'Test%');"
psql postgresql://postgres:postgres@localhost:5432/postgres -c \
  "DELETE FROM deals WHERE deal_name LIKE 'Test%';"
```
