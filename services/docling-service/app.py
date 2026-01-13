from fastapi import FastAPI, File, UploadFile, HTTPException
from docling.document_converter import DocumentConverter
from typing import Dict, List
import tempfile
import json
import os

app = FastAPI()

# Initialize converter (automatically detects GPU availability)
converter = DocumentConverter()

@app.post("/convert")
async def convert_document(file: UploadFile = File(...)) -> Dict:
    """
    Convert uploaded document to structured JSON.
    Supports: PDF, DOCX, PPTX, XLSX, images
    """
    tmp_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Convert document using Docling
        result = converter.convert(tmp_path)
        
        # Extract structured data
        doc = result.document
        
        pages_data = []
        for i, page in enumerate(doc.pages):
            page_data = {
                "page_number": i + 1,
                "text": page.export_to_text() if hasattr(page, 'export_to_text') else str(page),
                "markdown": page.export_to_markdown() if hasattr(page, 'export_to_markdown') else "",
                "tables": extract_tables(page) if hasattr(page, 'tables') else [],
            }
            pages_data.append(page_data)
        
        return {
            "success": True,
            "pages": pages_data,
            "metadata": {
                "page_count": len(doc.pages),
                "has_tables": any(hasattr(page, 'tables') and page.tables for page in doc.pages),
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

def extract_tables(page):
    """Extract table data with structure"""
    tables = []
    if not hasattr(page, 'tables'):
        return tables
    
    for table in page.tables:
        table_data = {
            "data": table.to_dict() if hasattr(table, 'to_dict') else str(table),
            "markdown": table.export_to_markdown() if hasattr(table, 'export_to_markdown') else str(table)
        }
        tables.append(table_data)
    return tables

@app.get("/health")
async def health():
    """Health check endpoint"""
    gpu_available = False
    try:
        # Try to detect GPU support
        gpu_available = hasattr(converter, 'has_gpu') and converter.has_gpu()
    except:
        pass
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "service": "docling-ocr"
    }

@app.get("/")
async def root():
    return {
        "service": "Docling OCR Service",
        "version": "1.0.0",
        "endpoints": {
            "convert": "POST /convert - Upload document for processing",
            "health": "GET /health - Check service health"
        }
    }


