from typing import List, Optional
from pydantic import BaseModel, Field
from typing import List


class BoundingBox(BaseModel):
    top_left: List[float] = Field(...,
                                  description="Top-left coordinates [x, y]")
    top_right: List[float] = Field(...,
                                   description="Top-right coordinates [x, y]")
    bottom_right: List[float] = Field(...,
                                      description="Bottom-right coordinates [x, y]")
    bottom_left: List[float] = Field(...,
                                     description="Bottom-left coordinates [x, y]")


class OCRResult(BaseModel):
    bbox: BoundingBox = Field(...,
                              description="Coordinates of the bounding box")
    text: str = Field(..., description="Detected text")
    confidence: Optional[float] = Field(None,
                                        description="Confidence score of the detection")

    class Config:
        json_schema_extra = {
            "example": {
                "bounding_box": {
                    "top_left": [819.0, 10.0],
                    "top_right": [948.0, 10.0],
                    "bottom_right": [948.0, 41.0],
                    "bottom_left": [819.0, 41.0]
                },
                "text": "Figure 2:",
                "confidence": 0.99928879737854
            }
        }


class OCRResponse(BaseModel):
    results: List[OCRResult] = Field(..., description="List of OCR results")


class TableOCRResponse(BaseModel):
    results: OCRResponse = Field(...,
                                 description="List of ocr results for each cell")
    html: str = Field(..., description="HTML representation of the table")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "bbox": {
                            "top_left": [10.0, 10.0],
                            "top_right": [100.0, 10.0],
                            "bottom_right": [100.0, 50.0],
                            "bottom_left": [10.0, 50.0]
                        },
                        "text": "Sample"
                    },
                    # ... more OCR results ...
                ],
                "html": "<table><tr><td>Sample</td><td>Table</td></tr></table>"
            }
        }
