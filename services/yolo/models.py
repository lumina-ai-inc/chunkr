from pydantic import BaseModel
from typing import List, Tuple, Optional

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
