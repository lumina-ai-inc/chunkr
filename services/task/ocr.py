from paddleocr import PaddleOCR
from pathlib import Path
from models import OCRResult
from typing import List
from multiprocessing import Pool, cpu_count


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


def process_image(args):
    ocr, image_path = args
    return ocr.ocr(str(image_path))[0]


def perform_paddle_ocr_batch(ocr: PaddleOCR, image_paths: List[Path]) -> list:
    num_processes = min(cpu_count(), len(image_paths))
    print(f"Performing OCR on {len(image_paths)} images with {num_processes} processes")
    with Pool(num_processes) as pool:
        raw_results = pool.map(
            process_image, [(ocr, path) for path in image_paths])

    ocr_results = [
        OCRResult(
            bounding_box=result[0],
            text=result[1][0],
            confidence=result[1][1]
        )
        for batch in raw_results
        for result in batch
    ]
    return ocr_results
