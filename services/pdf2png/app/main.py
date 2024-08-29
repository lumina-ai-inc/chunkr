from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Dict
import fitz

app = FastAPI()

class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float
    page_number: int

class ConversionResponse(BaseModel):
    png_pages: List[str]
    legend: Dict[str, Dict[str, str]]

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF to PNG converter API"}

@app.post("/convert", response_model=ConversionResponse)
async def convert_pdf_to_png(bounding_boxes: List[BoundingBox], file: UploadFile = File(...)):
    png_pages = []
    legend = {}
    with file.file as f:
        with fitz.open(stream=f.read()) as doc:
            for bounding_box in bounding_boxes:
                page = doc[bounding_box.page_number - 1]
                rect = fitz.Rect(bounding_box.left, bounding_box.top, bounding_box.left + bounding_box.width, bounding_box.top + bounding_box.height)
                page.show_pdf_page_as_image(clip=rect)
                png_name = f"page_{bounding_box.page_number}_{len(png_pages) + 1}.png"
                png_pages.append(png_name)
                legend[png_name] = {"file": file.filename, "page": bounding_box.page_number}
    return {"png_pages": png_pages, "legend": legend}