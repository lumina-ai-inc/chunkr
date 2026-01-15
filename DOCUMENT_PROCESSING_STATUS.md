# Document Processing Status & Solution

## Current Issue

**Documents are NOT being processed because:**
1. ❌ Deal routes (`/api/v1/deals/*`) are DISABLED in the backend
2. ❌ Upload endpoint doesn't exist - uploads never reach the server
3. ✅ Worker is running and waiting for messages
4. ✅ Redis queue is empty (no messages to process)
5. ✅ All services are running

## Root Cause

The deal routes were disabled because they use Diesel ORM which doesn't work with async `tokio-postgres`. They need conversion to raw SQL (like the conversation routes).

## Quick Solution Options

### Option 1: Convert Deal Routes to Raw SQL (Recommended)
**Time:** 2-3 hours  
**Benefits:** Permanent fix, follows existing patterns  
**Status:** Partially done (some routes already converted)

**What needs conversion:**
- `create_deal_route` - Create new deal
- `upload_deal_documents` - Upload files to S3 + queue to Redis  
- `get_deal_documents` - Fetch document status
- `get_deal_facts` - Fetch extracted facts

### Option 2: Use Mock Endpoints (Fastest - 30 mins)
**Time:** 30 minutes  
**Benefits:** Immediate testing, no backend changes  
**Downside:** Temporary, not production-ready

Create mock endpoints that:
1. Accept file uploads
2. Queue to Redis directly
3. Return mock responses

### Option 3: Re-enable Old Chunkr Task Routes
**Time:** 1 hour  
**Benefits:** Uses existing working code  
**Downside:** Uses old task-based flow, not deal-based

Use the existing `/api/v1/task` endpoint which works and processes documents.

## What's Already Working ✅

1. **Worker Infrastructure:**
   - ✅ `deal_document_worker` runs successfully
   - ✅ Connects to Redis, Postgres, MinIO
   - ✅ Listens to `deal_documents` queue

2. **Document Processing:**
   - ✅ CSV files - instant text extraction
   - ✅ Excel files (.xls/.xlsx) - `calamine` library
   - ✅ PDF files - Free PDFium extraction
   - ✅ PDF fallback - Azure OCR if configured
   - ✅ S3/MinIO storage integration

3. **Services:**
   - ✅ All Docker services running
   - ✅ Redis queue system
   - ✅ Postgres database
   - ✅ MinIO object storage

## Recommended Action Plan

### Phase 1: Quick Fix (Use Task Endpoints - 1 hour)

**Modify frontend to use existing `/api/v1/task` endpoint:**

```typescript
// In dealApi.ts
export const uploadDealDocuments = async (
  dealId: string,
  files: File[],
  documentType: string
): Promise<any> => {
  const formData = new FormData();
  formData.append("file", files[0]); // Task endpoint takes single file
  formData.append("ocr_strategy", "Auto");
  formData.append("segmentation_strategy", "LayoutAnalysis");
  
  const response = await axiosInstance.post("/api/v1/task", formData);
  return response.data;
};
```

### Phase 2: Proper Solution (Convert Deal Routes - 2-3 hours)

Convert deal routes from Diesel to raw SQL following the conversation routes pattern.

**Example (already working in `conversation.rs`):**

```rust
pub async fn create_deal_route(
    user_info: web::ReqData<UserInfo>,
    body: web::Json<CreateDealRequest>,
) -> Result<HttpResponse, Error> {
    let mut client = get_pg_client().await.map_err(|e| {
        eprintln!("Database connection error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection failed")
    })?;

    let deal_id = format!("deal-{}-{}", Utc::now().timestamp_millis(), uuid::Uuid::new_v4());
    let user_id = &user_info.user_id;
    let deal_name = &body.deal_name;

    let result = client.execute(
        "INSERT INTO deals (deal_id, user_id, deal_name, created_at, updated_at) 
         VALUES ($1, $2, $3, NOW(), NOW())",
        &[&deal_id, user_id, deal_name],
    ).await.map_err(|e| {
        eprintln!("Database insert error: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to create deal")
    })?;

    Ok(HttpResponse::Created().json(json!({
        "deal_id": deal_id,
        "deal_name": deal_name
    })))
}
```

## Files to Convert

1. `core/src/routes/deal.rs` - Main file (already has some raw SQL)
2. Lines to focus on:
   - Line 62-142: `create_deal_route` 
   - Line 174-320: `upload_deal_documents` ⚠️ CRITICAL
   - Line 322-358: `get_deal_documents`
   - Line 412-446: `get_deal_facts`

## Immediate Next Steps

**Choose ONE:**

### A) Quick Test (30 mins):
```bash
# 1. Modify frontend to use /api/v1/task endpoint
# 2. Upload a file
# 3. Check worker logs for processing
```

### B) Proper Fix (2-3 hours):
```bash
# 1. Convert deal.rs routes to raw SQL
# 2. Enable routes in lib.rs (already done)
# 3. Rebuild backend
# 4. Test uploads
```

### C) Use Chunkr's Original OCR (Recommended Long-term)

Based on https://github.com/lumina-ai-inc/chunkr, integrate:

1. **OCR Backend** (Docker service `ocr-backend`):
   - Already in docker-compose (disabled)
   - Uses `doctr` for free OCR
   - No Azure costs!

2. **Segmentation Backend** (Docker service `segmentation-backend`):
   - Layout analysis
   - Bounding box detection
   - Already in docker-compose (disabled)

**To enable Chunkr's free OCR:**

```yaml
# In docker-compose.local.yaml
segmentation-backend:
  deploy:
    replicas: 1  # Change from 0 to 1

ocr-backend:
  deploy:
    replicas: 1  # Change from 0 to 1
```

Then update `ocr_service.rs` to use `http://localhost:8002` instead of Azure.

## Summary

**Problem:** Deal routes are disabled → uploads don't work  
**Quick Fix:** Use existing `/api/v1/task` endpoint (1 hour)  
**Proper Fix:** Convert deal routes to raw SQL (2-3 hours)  
**Best Long-term:** Use Chunkr's free OCR backends (no Azure costs)

**Current Status:**
- ✅ Worker: Running and ready
- ✅ Services: All operational
- ✅ Processing: CSV, Excel, PDF support added
- ❌ Upload: Routes disabled, needs fix
- ⏳ OCR: Can use free Chunkr or paid Azure

**Recommendation:** Start with Quick Fix (Option A) to test immediately, then do Proper Fix (Option B) for production.


