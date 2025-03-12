import os
import cv2
import torch
import argparse
import uuid
import numpy as np
import asyncio
import uvicorn
import json
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
import gc

# Pydantic models matching server.py
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class BoundingBoxOutput(BaseModel):
    left: float
    top: float
    width: float 
    height: float

class InstanceOutput(BaseModel):
    boxes: list[BoundingBoxOutput]
    scores: list[float]
    classes: list[int]
    image_size: tuple[int, int]
    
class Instance(BaseModel):
    boxes: List[BoundingBox]
    scores: List[float]
    classes: List[int]
    image_size: Tuple[int, int]

class SerializablePrediction(BaseModel):
    instances: Instance

class FinalPrediction(BaseModel):
    instances: InstanceOutput

class OCRInput(BaseModel):
    bbox: BoundingBoxOutput
    text: str
    confidence: Optional[float] = None

# Global model variable
model = None

def map_yolo_to_segment_type(yolo_class: int, box_y: float, img_height: float) -> int:
    # Map YOLO classes to SegmentType from Rust code
    if yolo_class == 0:  # title
        return 10  # Title
    elif yolo_class == 1:  # plain text
        return 9   # Text
    elif yolo_class == 2:  # abandon
        # Check position to determine if header or footer
        relative_position = box_y / img_height
        if relative_position < 0.2:
            return 5  # PageHeader
        elif relative_position > 0.8:
            return 4  # PageFooter
        else:
            return 9  # Text (default if not clearly header/footer)
    elif yolo_class == 3:  # figure
        return 6   # Picture
    elif yolo_class == 4:  # figure_caption
        return 0   # Caption
    elif yolo_class == 5:  # table
        return 8   # Table
    elif yolo_class == 6:  # table_caption
        return 0   # Caption
    elif yolo_class == 7:  # table_footnote
        return 1   # Footnote
    elif yolo_class == 8:  # isolate_formula
        return 2   # Formula
    elif yolo_class == 9:  # formula_caption
        return 0   # Caption
    else:
        return 9   # Default to Text

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
    model = YOLOv10(filepath)
    print("Model loaded successfully!")
    
    yield
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

async def process_image(image_data: bytes, conf: float = 0.2, imgsz: int = 1024):
    temp_file = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_file, "wb") as f:
        f.write(image_data)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        det_res = model.predict(
            temp_file,
            imgsz=imgsz,
            conf=conf,
            device=device,
        )
        
        if det_res[0].boxes is not None:
            boxes = det_res[0].boxes.xyxy.cpu().numpy()
            cls = det_res[0].boxes.cls.cpu().numpy().astype(int)
            conf = det_res[0].boxes.conf.cpu().numpy()
            
            # Sort by y-coordinate for reading order
            order = np.argsort(boxes[:, 1])
            boxes = boxes[order]
            cls = cls[order]
            conf = conf[order]
            
            # Get image height for class mapping
            img_height = det_res[0].orig_shape[0]
            
            # Map YOLO classes to SegmentType
            mapped_classes = []
            for i, yolo_class in enumerate(cls):
                box_y = boxes[i, 1]  # Y coordinate for position-based mapping
                mapped_class = map_yolo_to_segment_type(yolo_class, box_y, img_height)
                mapped_classes.append(mapped_class)
            
            # Convert to Pydantic models
            bbox_list = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]) for box in boxes]
            bbox_output_list = [BoundingBoxOutput(
                left=box[0],
                top=box[1],
                width=box[2] - box[0],
                height=box[3] - box[1]
            ) for box in boxes]
            
            # Create instance objects with mapped classes
            instance = Instance(
                boxes=bbox_list,
                scores=conf.tolist(),
                classes=mapped_classes,
                image_size=det_res[0].orig_shape
            )
            
            instance_output = InstanceOutput(
                boxes=bbox_output_list,
                scores=conf.tolist(),
                classes=mapped_classes,
                image_size=det_res[0].orig_shape
            )
            
            serializable_pred = SerializablePrediction(instances=instance)
            final_pred = FinalPrediction(instances=instance_output)
            
            return serializable_pred, final_pred
        else:
            # Empty results
            return SerializablePrediction(
                instances=Instance(boxes=[], scores=[], classes=[], image_size=det_res[0].orig_shape)
            ), FinalPrediction(
                instances=InstanceOutput(boxes=[], scores=[], classes=[], image_size=det_res[0].orig_shape)
            )
    
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

@app.post("/batch_async", response_model=FinalPrediction)
async def create_od_task(
    file: UploadFile = File(...),
    ocr_data: str = Form(...)
):
    image_data = await file.read()
    _, final_pred = await process_image(image_data)
    return final_pred

@app.post("/batch")
async def batch_od(
    files: List[UploadFile] = File(...),
    ocr_data: str = Form(...)
):
    results = []
    for file in files:
        image_data = await file.read()
        _, final_pred = await process_image(image_data)
        results.append(final_pred)
    
    return results

@app.get("/")
async def root():
    return {"message": "YOLO DocLayout API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)