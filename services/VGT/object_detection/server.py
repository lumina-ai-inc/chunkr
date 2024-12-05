from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import cv2
import numpy as np
from pathlib import Path
from memory_inference import create_predictor, process_image_batch
import uvicorn
from fastapi import Body
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import json
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

# Add this model to define the expected structure

@app.post("/batch/")
async def process_image_batch_endpoint(
    files: List[UploadFile] = File(...),
    grid_dicts: str = Form(...)  # JSON string containing list of grid dicts
):    
    # Parse the JSON string into list of dictionaries
    grid_dicts = json.loads(grid_dicts)
    
    # Read all images
    images = []
    for file in files:
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        images.append(image)
    
    # Process the batch
    predictions, visualizations = process_image_batch(predictor, images, dataset_name="doclaynet", grid_dicts=grid_dicts)
    
    # Convert predictions to serializable format
    serializable_predictions = []
    for pred in predictions:
        serializable_pred = {
            "boxes": pred["instances"].pred_boxes.tensor.tolist(),
            "scores": pred["instances"].scores.tolist(),
            "classes": pred["instances"].pred_classes.tolist(),
            "image_size": [
                pred["instances"].image_size[0],
                pred["instances"].image_size[1]
            ]
        }
        serializable_predictions.append(serializable_pred)
    
    return serializable_predictions

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)