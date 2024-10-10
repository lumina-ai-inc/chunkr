import cv2
from pathlib import Path
from paddleocr import PaddleOCR, PPStructure
import time

from src.configs.ocr_config import OCR__MAX_SIZE

def get_ocr_engine() -> PaddleOCR:
    return PaddleOCR(use_angle_cls=True, lang="en", ocr_order_method="tb-xy", show_log=False)

def get_table_engine() -> PPStructure:
    # todo: add lang support
    return PPStructure(recovery=True, return_ocr_result_in_table=True, layout=False, structure_version="PP-StructureV2", show_log=False)


def calculate_slice_params(image_size, max_size) -> dict:
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

def perform_ocr(image_path: Path) -> list:
    timings = {}
    start_time = time.time()

    ocr = get_ocr_engine()
    timings['ocr_engine'] = time.time() - start_time

    max_size = OCR__MAX_SIZE
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    height, width = img.shape[:2]
    slice_params = calculate_slice_params((width, height), max_size)
    timings['slice_params'] = time.time() - start_time

    if slice_params:
        raw_results = ocr.ocr(str(image_path), det=True,
                              rec=True, cls=False, slice=slice_params)
    else:
        raw_results = ocr.ocr(str(image_path))
    
    timings['ocr_processing'] = time.time() - start_time

    if not raw_results or not raw_results[0]:
        return []

    print("OCR Timings: ", timings)

    return raw_results

def perform_ocr_table(image_path: Path) -> list:
    timings = {}
    start_time = time.time()

    table_engine = get_table_engine()
    timings['table_engine_init'] = time.time() - start_time

    img = cv2.imread(str(image_path))
    timings['image_read'] = time.time() - start_time

    result = table_engine(img)
    timings['table_ocr'] = time.time() - start_time

    for line in result:
        line.pop('img')
    timings['post_processing'] = time.time() - start_time

    print("Table OCR Timings: ", timings)

    return result
