from paddleocr import PaddleOCR
from pathlib import Path

def perform_paddle_ocr(ocr: PaddleOCR, image_path: Path) -> list:
    result = ocr.ocr(str(image_path))
    return result