from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict
import fitz
import base64
import json
import os
from datetime import datetime

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
    file: UploadFile = File(...)
):
    # Convert JSON string to list of BoundingBox
    bounding_boxes = [BoundingBox(**box) for box in json.loads(bounding_boxes)]
    png_pages = []
    with file.file as f:
        with fitz.open(stream=f.read(), filetype="pdf") as doc:
            for bounding_box in bounding_boxes:
                page = doc[bounding_box.page_number - 1]
                rect = fitz.Rect(bounding_box.left, bounding_box.top, bounding_box.left +
                                 bounding_box.width, bounding_box.top + bounding_box.height)
                pix = page.get_pixmap(clip=rect)
                png_data = pix.tobytes("png")
                base64_png = base64.b64encode(png_data).decode()
                png_pages.append(
                    {"bb_id": bounding_box.bb_id, "base64_png": base64_png})
    return {"png_pages": png_pages}


@app.post("/split")
async def split_pdf(
    file: UploadFile = File(...),
    pages_per_split: int = Form(...)
):
    split_pdfs = []
    with file.file as f:
        with fitz.open(stream=f.read(), filetype="pdf") as doc:
            total_pages = len(doc)
            for i in range(0, total_pages, pages_per_split):
                start_page = i
                end_page = min(i + pages_per_split, total_pages)
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=start_page,
                                   to_page=end_page - 1)
                pdf_bytes = new_doc.tobytes()
                base64_pdf = base64.b64encode(pdf_bytes).decode()
                split_pdfs.append({
                    "split_number": i//pages_per_split + 1,
                    "base64_pdf": base64_pdf
                })
                new_doc.close()
    return {"split_pdfs": split_pdfs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
