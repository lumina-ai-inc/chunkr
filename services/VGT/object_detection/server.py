from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
import numpy as np
from pathlib import Path
from memory_inference import create_predictor, process_image
import uvicorn
app = FastAPI()

# Global variables
predictor = None

# Configuration paths
CONFIG_FILE = "object_detection/configs/cascade/doclaynet_VGT_cascade_PTM.yaml"
WEIGHTS_PATH = "object_detection/weights/doclaynet_VGT_model.pth"

@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = create_predictor(CONFIG_FILE, WEIGHTS_PATH)
    print("Model loaded successfully!")

@app.post("/process-image/")
async def process_image_endpoint(file: UploadFile = File(...), grid_dict: dict = None):
    # Read the uploaded image file
    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Process the image
    predictions, result_image = process_image(predictor, image, dataset_name="doclaynet", grid_dict=grid_dict)

    # Save the processed image to a temporary location
    output_file = Path(f"/app/object_detection/tmp/{file.filename}_processed.jpg")
    cv2.imwrite(str(output_file), result_image)

    return predictions

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)