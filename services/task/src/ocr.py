from bs4 import BeautifulSoup
from pathlib import Path
import requests
from typing import List

from src.configs.task_config import TASK__OCR_SERVICE_URL
from src.models.ocr_model import OCRResult, OCRResponse, BoundingBox


def ppocr_raw(image_path: Path) -> list:
    return requests.post(f"{TASK__OCR_SERVICE_URL}/ocr", files={"image": image_path.read_bytes()}).json()


def ppocr(image_path: Path) -> List[OCRResult]:
    raw_results = ppocr_raw(image_path)

    ocr_results = []
    for result in raw_results[0]:
        if result and len(result) == 2 and result[0] and result[1]:
            polygon = result[0]
            ocr_results.append(
                OCRResult(
                    bbox=BoundingBox.calculate_bounding_box(polygon),
                    text=result[1][0],
                    confidence=result[1][1]
                )
            )

    return ocr_results


def ppstructure_table_raw(image_path: Path) -> list:
    return requests.post(f"{TASK__OCR_SERVICE_URL}/ocr/table", files={"image": image_path.read_bytes()}).json()


def ppstructure_table(image_path: Path) -> OCRResponse:
    raw_results = ppstructure_table_raw(image_path)

    if not raw_results or not raw_results[0]:
        return OCRResponse(results=[], html="")

    table_result = raw_results[0]

    cell_bbox_raw = table_result['res'].get('cell_bbox', [])
    html = table_result['res'].get('html', "")

    soup = BeautifulSoup(html, 'html.parser')
    cells = soup.find_all(['td', 'th'])

    ocr_result = []
    for bbox, cell in zip(cell_bbox_raw, cells):
        polygon = [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[4], bbox[5]],
            [bbox[6], bbox[7]]
        ]
        ocr_result.append(
            OCRResult(
                bbox=BoundingBox.calculate_bounding_box(polygon),
                text=cell.get_text(strip=True),
                confidence=None
            )
        )

    response = OCRResponse(results=ocr_result, html=html)
    return response
