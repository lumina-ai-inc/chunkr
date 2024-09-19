from pydantic import BaseModel, Field
from typing import List

class OCRResult(BaseModel):
    bounding_box: List[List[float]] = Field(..., description="Coordinates of the bounding box")
    text: str = Field(..., description="Detected text")
    confidence: float = Field(..., description="Confidence score of the detection")

    class Config:
        json_schema_extra = {
            "example": {
                "bounding_box": [
                    [819.0, 10.0],
                    [948.0, 10.0],
                    [948.0, 41.0],
                    [819.0, 41.0]
                ],
                "text": "Figure 2:",
                "confidence": 0.99928879737854
            }
        }

class OCRResponse(BaseModel):
    results: List[OCRResult] = Field(..., description="List of OCR results")