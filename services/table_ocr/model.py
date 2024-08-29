from pydantic import BaseModel
from typing import List, Dict
from enum import Enum

class OCRModel(str, Enum):
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Column(BaseModel):
    bounding_box: BoundingBox

class Cell(BaseModel):
    bounding_box: BoundingBox

class TableStructureResponse(BaseModel):
    data: Dict[int, List[str]]
    columns: List[Column]
    cells: List[Cell]

class OCRResult(BaseModel):
    data: Dict[int, List[str]]
