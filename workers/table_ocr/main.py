from fastapi import FastAPI, UploadFile, File, Form
from model import TableStructureResponse, OCRResult, OCRModel
from config import DEVICE
from ocr_service import process_table_image, ocr_table_image
from services import create_html_table

app = FastAPI()

@app.get("/readiness")
def readiness():
    return {"status": "ready"}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/table", response_model=TableStructureResponse)
async def process_table(file: UploadFile = File(...)):
    table_data = await process_table_image(file)
    return TableStructureResponse(data=table_data)

@app.post("/ocr/table", response_model=OCRResult)
async def ocr_table(
    file: UploadFile = File(...),
    ocr_model: OCRModel = Form(default=OCRModel.EASYOCR, description="OCR model to use")
):
    table_data = await ocr_table_image(file, ocr_model)
    return OCRResult(data=table_data)

@app.post("/html/table", response_model=str)
async def html_table(
    file: UploadFile = File(...),
    ocr_model: OCRModel = Form(default=OCRModel.EASYOCR, description="OCR model to use")
):
    table_data = await ocr_table_image(file, ocr_model)
    return create_html_table(table_data)

if __name__ == "__main__":
    import uvicorn
    print(DEVICE)
    uvicorn.run(app, host="0.0.0.0", port=8000)