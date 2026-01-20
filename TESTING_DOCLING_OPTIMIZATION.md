# Testing Guide: Docling Optimization

## Quick Test Checklist

### 1. Start Services

```bash
# Terminal 1: Docker services
sh dev.sh

# Terminal 2: Backend
cd core && cargo run

# Terminal 3: Frontend  
cd apps/web && npm run dev

# Browser
open http://localhost:5173
```

---

### 2. Test Each File Type

#### Test 1: PDF File (Baseline)
- ✅ Upload a PDF document
- ✅ Check console logs for: `✅ PDF file - no conversion needed`
- ✅ Verify "View Source" works immediately
- ✅ Verify facts are extracted

**Expected behavior:** Same as before (no change)

---

#### Test 2: DOCX File (Main Optimization)
- ✅ Upload a Word document (.docx)
- ✅ Check console logs for:
  ```
  📄 Sending DOCX file directly to Docling (optimized path)
  ✨ Skip PDF conversion - Docling handles original format natively
  ```
- ✅ Verify facts appear faster than before
- ✅ Check "View Source" generates PDF in background
- ✅ Verify tables are extracted correctly

**Expected behavior:** 
- Faster fact extraction (~5-10 sec instead of ~15 sec)
- Better table quality
- PDF available for viewing shortly after

---

#### Test 3: XLSX File (Spreadsheet)
- ✅ Upload an Excel file (.xlsx)
- ✅ Check console logs for Docling processing
- ✅ Verify numeric data and formulas extracted
- ✅ Check table structure preserved

**Expected behavior:** Native Excel parsing, better cell detection

---

#### Test 4: Image File (OCR)
- ✅ Upload a PNG/JPG with text
- ✅ Verify OCR text extracted
- ✅ Check PDF generated for viewing

**Expected behavior:** Better OCR quality from Docling

---

### 3. Backend Logs to Watch

Look for these messages in `cargo run` terminal:

```bash
# Good signs:
✅ PDF file - no conversion needed
📄 Sending DOCX file directly to Docling (optimized path)
✨ Skip PDF conversion - Docling handles original format natively
✅ PDF generated for viewing: <document_id>

# If you see these, something's wrong:
❌ "Failed to process document"
❌ "Docling service unreachable"
```

---

### 4. Frontend Logs to Watch

Open browser console (F12), look for:

```javascript
// Good:
"Document processing started"
"Facts extracted: 15"
"PDF URL available"

// Bad:
"Processing failed"
"Network error"
```

---

### 5. Performance Comparison

#### Before Optimization:
```
DOCX Upload → Convert to PDF (3-5s) → OCR (5-10s) → Facts (3-5s)
Total: ~15 seconds
```

#### After Optimization:
```
DOCX Upload → OCR directly (5-10s) → Facts (3-5s)
Total: ~10 seconds ✅

PDF generation happens in background (non-blocking)
```

**How to measure:**
1. Upload same DOCX file before/after changes
2. Time from "Upload" click to facts appearing
3. Should be ~30-50% faster

---

### 6. Test Data Samples

Create test files or use these:

**sample.docx** - Word document with:
- Text paragraphs
- Tables (2-3 columns)
- Images
- Headers/footers

**rent-roll.xlsx** - Spreadsheet with:
- Property names
- Rent amounts
- Unit numbers
- Date columns

**contract.pdf** - Already-PDF document:
- Multi-page
- Mixed text and tables

**property-photo.jpg** - Image with:
- Building address visible
- Clear text for OCR

---

### 7. Success Criteria

✅ **All file types process without errors**  
✅ **DOCX/XLSX processing is noticeably faster**  
✅ **Table extraction quality is better**  
✅ **PDF viewer still works (even if delayed)**  
✅ **No regression in PDF handling**  
✅ **Backend logs show optimization messages**

---

### 8. Troubleshooting

#### "Docling service unreachable"
```bash
# Check if Docling is running
docker ps | grep docling

# Restart if needed
docker restart data-extract-docling-service-1
```

#### "PDF generation failed"
```bash
# Check LibreOffice is available
which libreoffice

# Check ImageMagick
which convert

# Install if missing (macOS)
brew install libreoffice imagemagick
```

#### "Facts not appearing"
```bash
# Check Redis queue
docker exec data-extract-redis-1 redis-cli LLEN deal_documents
docker exec data-extract-redis-1 redis-cli LLEN fact_extraction

# Check worker logs
tail -f core/target/release/deal_document_worker.log
```

---

### 9. Advanced Testing

#### Load Test
```bash
# Upload 10 DOCX files simultaneously
# Check processing time doesn't degrade
```

#### Memory Usage
```bash
# Monitor backend memory
cargo build --release && /usr/bin/time -l ./target/release/core
```

#### Error Handling
```bash
# Upload corrupted DOCX
# Verify graceful failure
```

---

### 10. Rollback Test

If optimization causes issues:

```bash
# Revert changes
git log --oneline | head -5
git revert <commit-hash>

# Rebuild
make build-all

# Test again with old flow
```

---

## Quick Smoke Test (30 seconds)

```bash
1. Start all services (dev.sh, cargo run, npm run dev)
2. Upload report.docx
3. Check console for "Skip PDF conversion" message
4. Wait for facts to appear
5. Click "View Source"
6. ✅ If all works → Optimization successful!
```

---

## Reporting Issues

If you find bugs:

1. **Capture logs:**
   ```bash
   # Backend
   cargo run 2>&1 | tee backend.log
   
   # Browser console
   Right-click → Inspect → Console → Save log
   ```

2. **Note:**
   - File type and size
   - Error message
   - Expected vs actual behavior
   - Time to reproduce

3. **Check:**
   - S3 storage (both input/ and pdf/ paths exist?)
   - Database (documents table status?)
   - Redis queues (messages stuck?)

---

**Ready to test!** 🚀

Start with PDF (baseline), then DOCX (main improvement), then others.
