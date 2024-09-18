from paddleocr import PaddleOCR
from pathlib import Path

def perform_paddle_ocr(ocr: PaddleOCR, image_path: Path) -> list:
    try:
        image_path = str(image_path)
        result = ocr.ocr(image_path)
    except Exception as e:
        result = []
        print(f"An error occurred: {str(e)}")
    return result