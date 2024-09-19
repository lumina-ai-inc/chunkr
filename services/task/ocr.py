from paddleocr import PaddleOCR
from pathlib import Path
from models import OCRResult

def perform_paddle_ocr(ocr: PaddleOCR, image_path: Path) -> list:
    raw_results = ocr.ocr(str(image_path))
    ocr_results = [
        OCRResult(
            bounding_box=result[0],
            text=result[1][0],
            confidence=result[1][1]
        )
        for result in raw_results[0]
    ]
    return ocr_results
