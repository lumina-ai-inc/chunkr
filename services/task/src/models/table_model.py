from pydantic import BaseModel, Field
from typing import List, Optional

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Table(BaseModel):
    cell_bbox: List[List[float]]
    html: str
