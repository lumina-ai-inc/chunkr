import tempfile
import os
import base64
from pathlib import Path
from src.models.segment_model import Segment, SegmentType

def process_segment(segment, image_service, ocr_service, page_image_file_paths, segment_image_extension):
    segment.image = image_service.crop_image(
        page_image_file_paths[segment.page_number], segment.left, segment.top, segment.width, segment.height, segment_image_extension)
    segment_temp_file = tempfile.NamedTemporaryFile(
        suffix=f".{segment_image_extension}", delete=False)
    segment_temp_file.write(base64.b64decode(segment.image))
    segment_temp_file.close()
    try:
        if segment.segment_type == SegmentType.Table:
            segment.ocr = ocr_service.paddle_table(
                Path(segment_temp_file.name))
        else:
            segment.ocr = ocr_service.paddle_ocr(
                Path(segment_temp_file.name))
    finally:
        os.unlink(segment_temp_file.name)
    return segment