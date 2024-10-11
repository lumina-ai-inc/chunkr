import json
import fitz
import os

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

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
    bbox: BoundingBox = Field(..., description="Coordinates of the bounding box")
    text: str = Field(..., description="Detected text")
    confidence: Optional[float] = Field(None, description="Confidence score of the detection")


class OCRResponse(BaseModel):
    results: List[OCRResult] = Field(..., description="List of ocr results for each cell")
    html: Optional[str] = Field(None, description="HTML representation of the table")


class ProcessType(Enum):
    OCR = "ocr"
    SUMMARY = "summary"


def draw_bounding_boxes(pdf_path, json_data, output_path, draw_ocr=True):
    # Define colors for different types
    color_map = {
        "Caption": (1, 0, 0),  # Red
        "Footnote": (0, 1, 0),  # Green
        "Formula": (0, 0, 1),  # Blue
        "List item": (1, 1, 0),  # Yellow
        "Page footer": (1, 0.5, 0),  # Orange
        "Page header": (0.5, 0, 0.5),  # Purple
        "Picture": (1, 0.75, 0.8),  # Pink
        "Section header": (0.6, 0.3, 0),  # Brown
        "Table": (0.54, 0, 0),  # Dark red
        "Text": (0, 0, 0),  # Black
        "Title": (1, 0, 0),  # Red
    }

    # Load JSON data
    data = json_data

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Check if 'segments' key exists, if not use the data directly
        if any('segments' in item for item in data):
            page_segments = [seg for item in data for seg in item.get("segments", []) if seg["page_number"] == page_num + 1]
        else:
            page_segments = [item for item in data if item["page_number"] == page_num + 1]

        # Get page dimensions
        page_width = page.rect.width
        page_height = page.rect.height

        # Draw rectangles for each segment
        for seg in page_segments:
            # Create BoundingBox object
            bbox = BoundingBox(
                left=seg["bbox"]["left"],
                top=seg["bbox"]["top"],
                width=seg["bbox"]["width"],
                height=seg["bbox"]["height"]
            )

            # Scale coordinates according to input PDF
            segment_rect = fitz.Rect(
                bbox.left * page_width / seg["page_width"],
                bbox.top * page_height / seg["page_height"],
                (bbox.left + bbox.width) * page_width / seg["page_width"],
                (bbox.top + bbox.height) * page_height / seg["page_height"]
            )
            color = color_map.get(seg["segment_type"], (0, 0, 0))  # Default to black if type not found
            page.draw_rect(segment_rect, color=color, width=2)

            # Draw OCR bbox if available and draw_ocr is True
            if draw_ocr and seg.get("ocr"):
                for ocr_result in seg["ocr"]:
                    ocr_bbox = BoundingBox(**ocr_result["bbox"])
                    # Calculate absolute coordinates for OCR bbox
                    ocr_rect = fitz.Rect(
                        segment_rect.x0 + ocr_bbox.left * segment_rect.width,
                        segment_rect.y0 + ocr_bbox.top * segment_rect.height,
                        segment_rect.x0 + (ocr_bbox.left + ocr_bbox.width) * segment_rect.width,
                        segment_rect.y0 + (ocr_bbox.top + ocr_bbox.height) * segment_rect.height
                    )
                    page.draw_rect(ocr_rect, color=(0, 0.5, 0.5), width=1)  # Teal color for OCR boxes

    # Save the modified PDF
    pdf_document.save(output_path)
    pdf_document.close()

if __name__ == "__main__":
    json_path = "output/De Beers Jewellers Ltd 2021_json.json"
    if not os.path.exists(json_path):
        print(f"Error: The file {json_path} does not exist.")
        print("Please ensure the JSON file has been generated before running this script.")
        exit(1)

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    draw_bounding_boxes(
        "input/De Beers Jewellers Ltd 2021.pdf",
        json_data,
        "output/De Beers Jewellers Ltd 2021_Annotated.pdf",
    )