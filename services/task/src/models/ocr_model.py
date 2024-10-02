from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
import time

from src.configs.llm_config import LLM__MODEL, LLM__BASE_URL, LLM__INPUT_TOKEN_PRICE, LLM__OUTPUT_TOKEN_PRICE
from src.models.segment_model import Segment

class BoundingBox(BaseModel):
    top_left: List[float] = Field(...,
                                  description="Top-left coor dinates [x, y]")
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
    results: List[OCRResult] = Field(...,
                                 description="List of ocr results for each cell")
    html: Optional[str] = Field(None, description="HTML representation of the table")

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

class ProcessType(Enum):
    OCR = "ocr"
    SUMMARY = "summary"

class ProcessInfo(BaseModel):
    __start_time = time.time()
    segment_id: str = Field(..., description="ID of the segment")
    process_type: Optional[str] = Field(None, description="Type of the process")
    model_name: Optional[str] = LLM__MODEL or "paddleocr"
    base_url: Optional[str] = LLM__BASE_URL
    input_tokens: Optional[int] = Field(None, description="Number of input tokens")
    output_tokens: Optional[int] = Field(None, description="Number of output tokens")
    input_price: Optional[float] = LLM__INPUT_TOKEN_PRICE or 0
    output_price: Optional[float] = LLM__OUTPUT_TOKEN_PRICE or 0
    total_cost: Optional[float] = Field(None, description="Total cost of the process")
    detail: Optional[str] = Field(None, description="Additional details about the process")
    latency: Optional[float] = Field(None, description="Process latency in seconds")
    avg_ocr_confidence: Optional[float] = Field(None, description="Average OCR confidence score")

    def calculate_total_cost(self):
        self.total_cost = self.input_price * self.input_tokens + self.output_price * self.output_tokens

    def calculate_latency(self):
        self.latency = time.time() - self.__start_time

    def finalize(self):
        self.calculate_total_cost()
        self.calculate_latency()
