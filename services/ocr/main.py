from fastapi import FastAPI, UploadFile, File
import os
from pathlib import Path
import tempfile
import uvicorn

from src.ocr import perform_ocr, perform_ocr_table

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:    
        image_path = Path(temp_file.name)
        image_path.write_bytes(await image.read())
        try:
            result = perform_ocr(image_path)
        finally:
            os.unlink(image_path)
        return result

@app.post("/ocr/table")
async def ocr_table(image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:    
        image_path = Path(temp_file.name)
        image_path.write_bytes(await image.read())
        try:
            result = perform_ocr_table(image_path)
        finally:
            os.unlink(image_path)
        return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
