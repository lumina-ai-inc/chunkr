# Docling Optimization: Skip PDF Conversion for OCR

## Current Problem
The system converts ALL files (DOCX, XLSX, images) to PDF before sending to Docling, even though Docling natively supports these formats. This causes:
- **Slower processing** (2x file conversions)
- **Quality loss** (especially for DOCX tables)
- **Unnecessary LibreOffice dependencies**
- **Wasted compute resources**

## Proposed Solution
**Send original files directly to Docling, convert to PDF only for viewing**

---

## Architecture Changes

### Flow Comparison

#### BEFORE (Current):
```
Upload DOCX → Task.new → Convert to PDF (LibreOffice) → S3 (PDF) 
   → DealProcessor → Download PDF → Docling OCR → Extract Facts
```

#### AFTER (Optimized):
```
Upload DOCX → Task.new → S3 (original DOCX) → Async PDF Worker → S3 (PDF for viewer)
   ↓
DealProcessor → Download original DOCX → Docling OCR (native support) → Extract Facts
```

---

## Implementation Steps

### 1. Backend: Defer PDF Conversion
**File: `core/src/models/pipeline.rs`**
- **Current**: Lines 122-128 convert to PDF in `Pipeline::init`
- **Change**: Skip PDF conversion, use original file

**File: `core/src/models/task.rs`**
- **Current**: Uploads original file to `input_location`
- **Change**: No changes needed (already stores original)

### 2. Backend: Update Deal Processing
**File: `core/src/pipeline/deal_processing.rs`**
- **Current**: Lines 71-94 route based on extension
- **Change**: 
  - Remove special handling for PDF/XLSX/CSV
  - Send ALL files directly to Docling
  - Let Docling handle all formats natively

### 3. Backend: Create PDF Conversion Worker
**New File: `core/src/workers/pdf_conversion_worker.rs`**
- Listen to Redis queue `pdf_conversion`
- Convert original files to PDF for viewer
- Update task with `pdf_url` presigned URL
- Non-blocking (doesn't hold up OCR processing)

### 4. Backend: Update Docling Service Call
**File: `core/src/services/ocr_service.rs`**
- **Current**: Lines 144-159 send file to Docling
- **Change**: No changes needed! Already sends any file format

### 5. Database: Add PDF Conversion Status
**New Migration:**
```sql
ALTER TABLE tasks 
ADD COLUMN IF NOT EXISTS pdf_conversion_status VARCHAR(50) DEFAULT 'pending';
-- Values: 'pending', 'converting', 'completed', 'failed', 'not_needed'
```

### 6. Frontend: Handle Missing PDF URLs
**File: `apps/web/src/components/Documents/OCRDocumentViewer.tsx`**
- **Change**: Show "Generating PDF preview..." if `pdf_url` is null
- Poll for PDF conversion status
- Display original filename in viewer

---

## Benefits

### Performance Improvements
- **50% faster** OCR processing (skip unnecessary conversion)
- **Better quality** extraction (Docling sees original formatting)
- **Parallel processing** (OCR and PDF conversion run simultaneously)

### Code Simplification
- Remove CSV/Excel special handling in `deal_processing.rs`
- Unified code path for all document types
- Docling handles file type detection

### User Experience
- Faster document analysis (facts appear sooner)
- PDF preview loads in background
- No blocking on LibreOffice conversion

---

## File Types Supported

| Format | Before (Converted to PDF) | After (Native) | Benefit |
|--------|---------------------------|----------------|---------|
| PDF    | ✅ (no conversion)        | ✅ (no conversion) | Same |
| DOCX   | ⚠️ LibreOffice → PDF     | ✅ Native Docling | Better tables, faster |
| XLSX   | ⚠️ Calamine extraction   | ✅ Native Docling | Better formatting |
| PPTX   | ⚠️ LibreOffice → PDF     | ✅ Native Docling | Better slides |
| Images | ⚠️ ImageMagick → PDF     | ✅ Native Docling | Better OCR |
| CSV    | ⚠️ Manual parsing        | ✅ Native Docling | Unified handling |

---

## Rollback Plan

If issues arise, the optimization can be easily rolled back:
1. Revert `pipeline.rs` to convert to PDF
2. Disable PDF conversion worker
3. Revert `deal_processing.rs` to use `process_pdf`, `process_excel`, etc.

All existing data and S3 storage patterns remain unchanged.

---

## Testing Checklist

- [ ] Upload PDF → OCR works, view works
- [ ] Upload DOCX → OCR works, PDF generates async, view works
- [ ] Upload XLSX → OCR works, tables extracted
- [ ] Upload PNG → OCR works
- [ ] Upload PPTX → OCR works
- [ ] Check S3 storage (input_location + pdf_location)
- [ ] Check async PDF conversion completes
- [ ] Check frontend handles missing pdf_url gracefully

---

## Implementation Order

1. ✅ Create this plan document
2. ⏳ Update `deal_processing.rs` to send originals to Docling
3. ⏳ Update `pipeline.rs` to skip PDF conversion
4. ⏳ Create PDF conversion worker
5. ⏳ Add database migration for `pdf_conversion_status`
6. ⏳ Update frontend to handle async PDF generation
7. ⏳ Test all file formats
8. ⏳ Deploy and monitor

---

## Code Changes Summary

### Files to Modify:
1. `core/src/pipeline/deal_processing.rs` - Remove file-type routing, use Docling for all
2. `core/src/models/pipeline.rs` - Skip PDF conversion in `init()`
3. `core/src/workers/pdf_conversion_worker.rs` - **NEW FILE** for async conversion
4. `core/migrations/XXXX_add_pdf_conversion_status/up.sql` - **NEW FILE**
5. `apps/web/src/components/Documents/OCRDocumentViewer.tsx` - Handle missing PDFs

### Files That DON'T Need Changes:
- ✅ `services/docling-service/app.py` - Already supports all formats
- ✅ `core/src/services/ocr_service.rs` - Already sends files correctly
- ✅ `core/src/models/task.rs` - Already stores originals

---

**Estimated Time**: 2-3 hours
**Risk Level**: Low (easy rollback)
**Impact**: High (major performance improvement)
