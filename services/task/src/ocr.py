from bs4 import BeautifulSoup
import cv2
from paddleocr import PaddleOCR, PPStructure
from pathlib import Path
from typing import List
from src.configs.task_config import TASK__OCR_MAX_SIZE
from src.models.ocr_model import OCRResult, OCRResponse, BoundingBox


def get_ocr():
    return PaddleOCR(use_angle_cls=True, lang="en",
                     ocr_order_method="tb-xy", show_log=False)


def get_table_engine():
    # todo: add lang support
    return PPStructure(
        recovery=True, return_ocr_result_in_table=True, layout=False, structure_version="PP-StructureV2", show_log=False)


def calculate_slice_params(image_size, max_size):
    width, height = image_size
    if width <= max_size and height <= max_size:
        return None

    h_stride = max(300, width // 8)
    v_stride = max(300, height // 8)

    merge_x_thres = max(20, h_stride // 10)
    merge_y_thres = max(20, v_stride // 10)

    return {
        'horizontal_stride': h_stride,
        'vertical_stride': v_stride,
        'merge_x_thres': merge_x_thres,
        'merge_y_thres': merge_y_thres
    }


def ppocr_raw(image_path: Path) -> list:
    ocr = get_ocr()
    return ocr.ocr(str(image_path))


def ppocr(image_path: Path) -> List[OCRResult]:
    ocr = get_ocr()

    max_size = TASK__OCR_MAX_SIZE
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    height, width = img.shape[:2]
    slice_params = calculate_slice_params((width, height), max_size)

    if slice_params:
        raw_results = ocr.ocr(str(image_path), det=True,
                              rec=True, cls=False, slice=slice_params)
    else:
        raw_results = ocr.ocr(str(image_path))

    if not raw_results or not raw_results[0]:
        return []

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
    table_engine = get_table_engine()

    img = cv2.imread(str(image_path))
    result = table_engine(img)
    for line in result:
        line.pop('img')
    return result


def ppstructure_table(image_path: Path) -> OCRResponse:
    table_engine = get_table_engine()

    img = cv2.imread(str(image_path))
    result = table_engine(img)

    table_result = result[0] if result else None

    if not table_result:
        return OCRResponse(results=[], html="")

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
