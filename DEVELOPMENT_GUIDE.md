# ReFlow Development Guide

## Quick Start (First Time Setup)

### Prerequisites
- Docker Desktop installed and running
- Rust toolchain (`rustup`)
- Node.js 18+ and npm
- PostgreSQL client (optional, for debugging)
- **LibreOffice + ImageMagick** (for document preview - see below)

### Initial Setup

```bash
# 1. Clone and navigate to project
cd /path/to/data-extract

# 2. Install PDF conversion tools (required for DOCX/XLSX preview)
./scripts/setup-pdf-conversion.sh

# 3. Copy environment file
#cp .env.example .env  # Edit with your API keys

# 4. Build everything (first time only)
make build-all
```

**What `make build-all` does:**
- Compiles Rust backend (release mode)
- Builds all worker binaries
- Compiles TypeScript frontend
- Takes ~5-10 minutes first time

**Why PDF conversion tools?**
- Browsers can only preview PDFs inline (not DOCX/XLSX)
- The worker converts DOCX/XLSX → PDF for viewing
- Without LibreOffice: documents download instead of previewing
- The setup script installs: `libreoffice` + `imagemagick`

---

## Daily Development Workflow

### Step 1: Start Services (Once per day)

```bash
./dev.sh
```

This starts Docker services in background:
- ✅ PostgreSQL (database)
- ✅ Redis (queue)
- ✅ MinIO (file storage)
- ✅ Qdrant (vector DB)
- ✅ Docling (OCR service)

### Step 2: Run Application (3 terminals)

**Terminal 1: Backend API**
```bash
cd core
cargo run
```
- Starts on `http://localhost:8000`
- Hot-reload: Just Ctrl+C and re-run after code changes
- Build time: ~30 seconds

**Terminal 2: Workers**
```bash
./scripts/run-all-workers-local.sh
```
- Runs document processing and fact extraction workers
- Listens to Redis queues
- Restart after worker code changes

**Terminal 3: Frontend**
```bash
cd apps/web
npm run dev
```
- Starts on `http://localhost:5173`
- Hot-reload: Automatic on file save
- Build time: ~5 seconds

---

## Development Tips

### When to Rebuild

**Full rebuild (`make build-all`):**
- ❌ Not needed for daily development
- ✅ Only after pulling major changes
- ✅ After changing dependencies (Cargo.toml, package.json)

**Partial rebuilds:**
```bash
# Backend only (if Rust code changed)
cd core && cargo build --release

# Frontend only (if React code changed)
cd apps/web && npm run build

# Workers only
cd core && cargo build --release --bin deal_document_worker --bin fact_extraction_worker
```

### Fast Iteration Cycle

1. **Backend changes:** Ctrl+C in Terminal 1, then `cargo run` again
2. **Frontend changes:** Just save file (auto-reloads)
3. **Worker changes:** Ctrl+C in Terminal 2, then `./scripts/run-all-workers-local.sh` again

### Checking Service Health

```bash
# Check Docker services
docker compose -f docker-compose.local.yaml ps

# Check Docling OCR
curl http://localhost:8002/health

# Check Backend API
curl http://localhost:8000/health

# Check Redis queue
redis-cli LLEN deal_documents
```

---

## Testing Document Processing

### Upload a Test Document

1. Open `http://localhost:5173`
2. Click "+ New Deal"
3. Enter deal name: "Test Property"
4. Upload a PDF (rent roll, P&L, etc.)
5. Watch Terminal 2 (workers) for processing logs

### Expected Flow

```
Frontend Upload
    ↓
Backend API (Terminal 1)
    ↓
Redis Queue
    ↓
Deal Worker (Terminal 2) → Docling OCR
    ↓
Store OCR Results
    ↓
Fact Extraction Worker (Terminal 2) → AI Agent
    ↓
Store Facts
    ↓
Frontend displays facts
```

### Monitor Processing

**Terminal 2 output:**
```
[METRIC] {"metric":"s3_download_duration_ms",...}
[METRIC] {"metric":"ocr_duration_ms",...}
[METRIC] {"metric":"pipeline_complete",...}
Fact extraction completed: X facts extracted
```

---

## Troubleshooting

### Issue: "Connection refused" errors

**Solution:** Make sure Docker services are running
```bash
docker compose -f docker-compose.local.yaml ps
# If not running:
./dev.sh
```

### Issue: "Port already in use"

**Solution:** Kill existing processes
```bash
# Find process on port 8000 (backend)
lsof -ti:8000 | xargs kill -9

# Find process on port 5173 (frontend)
lsof -ti:5173 | xargs kill -9
```

### Issue: Workers not processing documents

**Solution:** Check Redis connection
```bash
redis-cli ping  # Should return PONG
redis-cli LLEN deal_documents  # Check queue length
```

### Issue: Docling service not responding

**Solution:** Restart Docling
```bash
docker compose -f docker-compose.local.yaml restart docling-service
curl http://localhost:8002/health  # Should return {"status":"healthy"}
```

### Issue: Database errors

**Solution:** Reset database
```bash
docker compose -f docker-compose.local.yaml down -v
docker compose -f docker-compose.local.yaml up -d postgres
cd core && diesel migration run
```

---

## Stopping Development

### Stop Application (keep Docker running)
- Ctrl+C in all 3 terminals

### Stop Everything (including Docker)
```bash
docker compose -f docker-compose.local.yaml down
```

### Clean Restart
```bash
# Stop and remove all data
docker compose -f docker-compose.local.yaml down -v

# Start fresh
./dev.sh
# Then run your 3 terminals again
```

---

## Environment Variables

Key variables in `.env`:

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
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chunkr

# Redis
REDIS__URL=redis://localhost:6379
```

---

## Performance Benchmarks

**Expected processing times:**
- S3 Download: < 1 second
- OCR Processing: < 10 seconds (typical PDF)
- Fact Extraction: < 5 seconds
- **Total**: < 30 seconds end-to-end

**If slower:**
- Check Docling service: `docker stats docling-service`
- Check worker logs in Terminal 2
- Verify network connectivity

---

## Production Deployment

For production, use Docker Compose:

```bash
# Build production images
docker compose -f docker-compose.production.yaml build

# Deploy
docker compose -f docker-compose.production.yaml up -d
```

All services run in Docker (no local terminals needed).

---

## Additional Resources

- **Architecture:** See `IMPLEMENTATION_COMPLETE.md`
- **Testing:** See `TESTING_DOCUMENT_PROCESSING.md`
- **API Docs:** `http://localhost:8000/docs` (when backend running)
- **Troubleshooting:** Check worker logs in Terminal 2

---

## Summary: Daily Workflow

```bash
# Morning (once)
./dev.sh

# Then open 3 terminals:
# Terminal 1: cd core && cargo run
# Terminal 2: ./scripts/run-all-workers-local.sh
# Terminal 3: cd apps/web && npm run dev

# Work on code, save files, test

# Evening
# Ctrl+C in all terminals
# docker compose -f docker-compose.local.yaml down
```

**That's it! Happy coding! 🚀**
