from paddleocr import PaddleOCR
from pathlib import Path
from more_itertools import split_when

def get_y_center(box):
    return (box[0][0][1] + box[0][2][1]) / 2

def not_vertically_overlapping(box1, box2):
    up1, down1 = box1[0][0][1], box1[0][2][1]
    up2, down2 = box2[0][0][1], box2[0][2][1]
    return down1 < up2 or (down1 - up2) < (up2 - up1) / 2

def group_by_row(ocr_result):
    sorted_boxes = sorted(ocr_result, key=get_y_center)
    rows = list(split_when(sorted_boxes, not_vertically_overlapping))
    return [sorted(row, key=lambda x: x[0][0][0]) for row in rows]

def perform_paddle_ocr(ocr: PaddleOCR, image_path: Path) -> list:
    result = ocr.ocr(str(image_path))
    grouped_result = group_by_row(result[0])
    return grouped_result