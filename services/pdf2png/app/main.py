from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict
import fitz
import base64
import json

app = FastAPI()

class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int

class ConversionResponse(BaseModel):
    png_pages: List[Dict[str, str]]

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF to PNG converter API"}

@app.post("/convert", response_model=ConversionResponse)
async def convert_pdf_to_png(
    bounding_boxes: str = Form(...),  # Receive bounding_boxes as a JSON string
    file: UploadFile = File(...)
):
    bounding_boxes = [BoundingBox(**box) for box in json.loads(bounding_boxes)]  # Convert JSON string to list of BoundingBox
    png_pages = []
    with file.file as f:
        with fitz.open(stream=f.read(), filetype="pdf") as doc:
            for bounding_box in bounding_boxes:
                page = doc[bounding_box.page_number - 1]
                rect = fitz.Rect(bounding_box.left, bounding_box.top, bounding_box.left + bounding_box.width, bounding_box.top + bounding_box.height)
                pix = page.get_pixmap(clip=rect)
                png_data = pix.tobytes("png")
                base64_png = base64.b64encode(png_data).decode()
                png_pages.append({"bounding_box": json.dumps(bounding_box.dict()), "base64_png": base64_png})
    return {"png_pages": png_pages}