# OCR Processing Fix & CSV Support

## Issues Fixed

### 1. OCR Processing Stuck at "⏳ Processing Document"

**Root Cause:** The `deal_document_worker` was created but never started. Documents were being queued to Redis (`deal_documents` queue) but no worker was consuming them.

**Solution:**
- Added `deal_document_worker` binary to `core/Cargo.toml`
- Created `docker/deal-worker/Dockerfile` for containerized deployment
- Added `deal-worker` service to `docker-compose.production.yaml` and `docker-compose.local.yaml`
- Worker now listens to the `deal_documents` Redis queue and processes documents using Azure Document Intelligence

### 2. CSV File Support Added

**Root Cause:** The system only supported PDF files for OCR processing. CSV files (like rent rolls) are critical for real estate analysis but were not supported.

**Solution:**
- Added CSV detection in `core/src/pipeline/deal_processing.rs`
- CSV files bypass OCR and are parsed directly as text
- CSV content is stored in the same OCR output format for consistency
- Supports both PDF (via Azure OCR) and CSV (direct text parsing)

## Files Modified

### Backend
1. **`core/Cargo.toml`** - Added `deal_document_worker` binary
2. **`core/src/workers/deal_document_worker.rs`** - Fixed module imports
3. **`core/src/pipeline/deal_processing.rs`** - Added CSV support and file type detection
4. **`docker-compose.production.yaml`** - Added `deal-worker` service
5. **`docker-compose.local.yaml`** - Added local override for `deal-worker`
6. **`docker/deal-worker/Dockerfile`** - New Dockerfile for worker

### Scripts
7. **`scripts/run-deal-worker-local.sh`** - Script to run worker locally for testing

### Documentation
8. **`OCR_PROCESSING_FIX.md`** - This file

## How to Test

### Option 1: Run Worker Locally (Fastest for Development)

```bash
# 1. Make sure services are running
make start

# 2. In a new terminal, run the worker
./scripts/run-deal-worker-local.sh
```

The worker will:
- Connect to Redis and listen for `deal_documents` queue
- Process any queued documents
- Print status messages to console
- Continue running until you press Ctrl+C

### Option 2: Run Worker in Docker (Production-like)

```bash
# 1. Build the worker image
docker build -f docker/deal-worker/Dockerfile -t luminainc/deal_document_worker:1.20.1 .

# 2. Start all services (including the worker)
make start

# 3. Check worker logs
docker compose -f docker-compose.production.yaml -f docker-compose.local.yaml logs -f deal-worker
```

### Testing Document Upload

1. **Upload a PDF document:**
   - Go to Dashboard → "+ New Deal"
   - Upload a PDF (e.g., rent roll, P&L statement)
   - Watch the processing status change from "⏳ Processing" to "✅ Complete"

2. **Upload a CSV file:**
   - Upload a CSV rent roll
   - Should process instantly (no OCR needed)
   - Content will be available in the Facts tab

3. **Check worker logs:**
   ```bash
   # If running locally
   # See output in the terminal where you ran run-deal-worker-local.sh
   
   # If running in Docker
   docker compose logs -f deal-worker
   ```

Expected log output:
```
Deal Document Worker starting...
Listening for deal documents on queue: deal_documents
Received document processing message: {"document_id":"...","deal_id":"...","user_id":"...","s3_location":"...","file_name":"...","document_type":"..."}
Processing document doc-123 for deal deal-456
Downloaded document from S3: s3://bucket/user/deal/file.pdf
Using OCR service: Azure Document Intelligence
Processing completed: 5 pages processed
OCR results stored for document doc-123
Fact extraction triggered for document doc-123
Document processing completed: doc-123
```

## Architecture Overview

```
User Upload → API Server → S3 Storage + Redis Queue
                                          ↓
                                    deal_document_worker
                                          ↓
                          ┌───────────────┴───────────────┐
                          │                               │
                     PDF Files                       CSV Files
                          │                               │
                   Azure OCR API                   Direct Parse
                          │                               │
                          └───────────────┬───────────────┘
                                          ↓
                                   Store in Postgres
                                          ↓
                              Trigger Fact Extraction
                                          ↓
                                   Update UI Status
```

## Environment Variables Required

Make sure these are set in your `.env` file:

```bash
# Azure Document Intelligence (for PDF OCR)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-api-key

# Redis (for queue)
REDIS_URL=redis://localhost:6379

# PostgreSQL (for storage)
DATABASE_URL=postgresql://user:pass@localhost:5432/orin

# S3/MinIO (for file storage)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_S3_ENDPOINT=http://localhost:9000
AWS_S3_BUCKET=orin-documents
```

## Troubleshooting

### Worker Not Processing Documents

1. **Check if worker is running:**
   ```bash
   docker compose ps | grep deal-worker
   # or
   ps aux | grep deal_document_worker
   ```

2. **Check Redis queue:**
   ```bash
   docker exec -it data-extract-redis-1 redis-cli
   > LLEN deal_documents
   > LRANGE deal_documents 0 -1
   ```

3. **Check worker logs:**
   ```bash
   docker compose logs -f deal-worker
   ```

### Azure OCR Errors

1. **Verify credentials:**
   ```bash
   echo $AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
   echo $AZURE_DOCUMENT_INTELLIGENCE_KEY
   ```

2. **Test Azure connection:**
   ```bash
   curl -H "Ocp-Apim-Subscription-Key: $AZURE_DOCUMENT_INTELLIGENCE_KEY" \
        "$AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT/documentintelligence/documentModels?api-version=2024-07-31-preview"
   ```

### CSV Files Not Processing

1. **Check file extension:** File must have `.csv` extension
2. **Check file encoding:** Should be UTF-8
3. **Check worker logs:** Look for "Processing CSV file" message

## Next Steps

1. **Monitor Performance:**
   - Track processing times for different file sizes
   - Monitor Azure API usage and costs
   - Adjust `deal-worker` replicas based on load

2. **Add More File Types:**
   - Excel files (.xlsx, .xls)
   - Images (.jpg, .png) for property photos
   - Word documents (.docx) for appraisal reports

3. **Improve Error Handling:**
   - Retry failed OCR attempts
   - Better error messages for users
   - Dead letter queue for permanently failed documents

4. **Add Progress Updates:**
   - Real-time progress for multi-page documents
   - Estimated completion time
   - Page-by-page results streaming

