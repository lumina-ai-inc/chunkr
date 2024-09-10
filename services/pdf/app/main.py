from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict
import base64
import json
import io
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_bytes
import math

app = FastAPI()

class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int
    bb_id: str

class ConversionResponse(BaseModel):
    png_pages: List[Dict[str, str]]

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF to PNG converter API"}

@app.post("/convert", response_model=ConversionResponse)
async def convert_pdf_to_png(
    bounding_boxes: str = Form(...),  # Receive bounding_boxes as a JSON string
    file: UploadFile = File(...),
    dpi: int = Form(150)
):
    # Convert JSON string to list of BoundingBox
    bounding_boxes = [BoundingBox(**box) for box in json.loads(bounding_boxes)]
    png_pages = []
    
    pdf_bytes = await file.read()
    
    # Get PDF dimensions and DPI
    pdf = PdfReader(io.BytesIO(pdf_bytes))
    first_page = pdf.pages[0]
    pdf_width = float(first_page.mediabox.width)
    pdf_height = float(first_page.mediabox.height)
    
    # Convert PDF to images
    pdf_images = convert_from_bytes(pdf_bytes, dpi=dpi) 
    
    for bounding_box in bounding_boxes:
        # PDF page numbers start from 1, but list indexing starts from 0
        img = pdf_images[bounding_box.page_number - 1]
        
        # Calculate scaling factor
        scale_x = img.width / pdf_width
        scale_y = img.height / pdf_height
        
        # Adjust bounding box coordinates
        left = math.floor(bounding_box.left * scale_x)
        top = math.floor(bounding_box.top * scale_y)
        right = math.ceil((bounding_box.left + bounding_box.width) * scale_x)
        bottom = math.ceil((bounding_box.top + bounding_box.height) * scale_y)
        
        # Crop image based on adjusted bounding box
        cropped_img = img.crop((left, top, right, bottom))
        
        # Convert to PNG
        buffer = io.BytesIO()
        cropped_img.save(buffer, format="PNG")
        png_data = buffer.getvalue()
        base64_png = base64.b64encode(png_data).decode()
        
        png_pages.append(
            {"bb_id": bounding_box.bb_id, "base64_png": base64_png}
        )
    
    return {"png_pages": png_pages}

@app.post("/split")
async def split_pdf(
    file: UploadFile = File(...),
    pages_per_split: int = Form(...)
):
    split_pdfs = []
    pdf_bytes = await file.read()
    
    with io.BytesIO(pdf_bytes) as pdf_file:
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        
        for i in range(0, total_pages, pages_per_split):
            start_page = i
            end_page = min(i + pages_per_split, total_pages)
            
            writer = PdfWriter()
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
            
            output = io.BytesIO()
            writer.write(output)
            output.seek(0)
            
            base64_pdf = base64.b64encode(output.getvalue()).decode()
            split_pdfs.append({
                "split_number": i//pages_per_split + 1,
                "base64_pdf": base64_pdf
            })
    
    return {"split_pdfs": split_pdfs}

