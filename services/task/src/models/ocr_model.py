from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
import time

from src.configs.llm_config import LLM__MODEL, LLM__BASE_URL, LLM__INPUT_TOKEN_PRICE, LLM__OUTPUT_TOKEN_PRICE


class BoundingBox(BaseModel):
    left: float = Field(..., description="Left coordinate of the bounding box")
    top: float = Field(..., description="Top coordinate of the bounding box")
    width: float = Field(..., description="Width of the bounding box")
    height: float = Field(..., description="Height of the bounding box")

    @classmethod
    def calculate_bounding_box(cls, polygon: List[List[float]]) -> 'BoundingBox':
        """
        Calculate the largest bounding box that can fit the given polygon.

        :param polygon: A list of [x, y] coordinates representing the polygon
        :return: A BoundingBox object
        """
        x_coordinates = [point[0] for point in polygon]
        y_coordinates = [point[1] for point in polygon]

        left = min(x_coordinates)
        top = min(y_coordinates)
        right = max(x_coordinates)
        bottom = max(y_coordinates)

        width = right - left
        height = bottom - top

        return cls(left=left, top=top, width=width, height=height)


class OCRResult(BaseModel):
    bbox: BoundingBox = Field(...,
                              description="Coordinates of the bounding box")
    text: str = Field(..., description="Detected text")
    confidence: Optional[float] = Field(None,
                                        description="Confidence score of the detection")


class OCRResponse(BaseModel):
    results: List[OCRResult] = Field(...,
                                     description="List of ocr results for each cell")
    html: Optional[str] = Field(
        None, description="HTML representation of the table")


class ProcessType(Enum):
    OCR = "ocr"
    SUMMARY = "summary"


class ProcessInfo(BaseModel):
    __start_time = time.time()
    segment_id: str = Field(..., description="ID of the segment")
    process_type: Optional[str] = Field(
        None, description="Type of the process")
    llm_model_name: Optional[str] = LLM__MODEL or "paddleocr"
    base_url: Optional[str] = LLM__BASE_URL
    input_tokens: Optional[int] = Field(
        0, description="Number of input tokens")
    output_tokens: Optional[int] = Field(
        0, description="Number of output tokens")
    input_price: Optional[float] = LLM__INPUT_TOKEN_PRICE or 0
    output_price: Optional[float] = LLM__OUTPUT_TOKEN_PRICE or 0
    total_cost: Optional[float] = Field(
        None, description="Total cost of the process")
    detail: Optional[str] = Field(
        None, description="Additional details about the process")
    latency: Optional[float] = Field(
        None, description="Process latency in seconds")
    avg_ocr_confidence: Optional[float] = Field(
        None, description="Average OCR confidence score")

    def calculate_total_cost(self):
        self.total_cost = self.input_price * self.input_tokens + \
            self.output_price * self.output_tokens

    def calculate_latency(self):
        self.latency = time.time() - self.__start_time

    def finalize(self):
        self.calculate_total_cost()
        self.calculate_latency()
