from fastapi import FastAPI, UploadFile, File, Form
from config import DEVICE
from ocr_service import process_table_image

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

@app.post("/predict/table")
async def process_table(file: UploadFile = File(...)):
    table_data = await process_table_image(file)
    print(table_data)
    return table_data



if __name__ == "__main__":
    import uvicorn
    print(DEVICE)
    uvicorn.run(app, host="0.0.0.0", port=8000)