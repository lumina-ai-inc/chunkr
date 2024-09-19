import tempfile
import cv2
import pandas as pd
from paddleocr import PaddleOCR, PPStructure, save_structure_res
from pathlib import Path
from openpyxl import load_workbook, Workbook
from openpyxl.drawing.image import Image as XLImage
from pprint import pprint
import os

from src.models.ocr_model import OCRResult, OCRResponse


def perform_paddle_ocr(ocr: PaddleOCR, image_path: Path) -> OCRResponse:
    raw_results = ocr.ocr(str(image_path))
    ocr_results = [
        OCRResult(
            bounding_box=result[0],
            text=result[1][0],
            confidence=result[1][1]
        )
        for result in raw_results[0]
    ]
    return OCRResponse(results=ocr_results)


def ppstructure_table(table_engine: PPStructure, image_path: Path) -> Path:
    img = cv2.imread(str(image_path))
    result = table_engine(img)
    for line in result:
        line.pop('img')
        print(line)
    pprint(result)
    return result
