import base64
import os
import tempfile
import threading
from paddleocr import PaddleOCR, PPStructure
from pathlib import Path

from src.converters import crop_image
from src.models.segment_model import BaseSegment, Segment, SegmentType
from src.ocr import ppocr, ppstructure_table


def adjust_base_segments(segments: list[BaseSegment], offset: float = 5.0, density: int = 300, pdla_density: int = 72):
    scale_factor = density / pdla_density
    for segment in segments:
        # Scale dimensions and positions
        segment.width *= scale_factor
        segment.height *= scale_factor
        segment.left *= scale_factor
        segment.top *= scale_factor

        # Apply offset
        segment.width += offset * 2
        segment.height += offset * 2
        segment.left -= offset
        segment.top -= offset


def process_segment_ocr(
    segment: Segment,
    segment_temp_file: str,
    ocr: PaddleOCR,
    table_engine: PPStructure,
    ocr_lock: threading.Lock,
    table_engine_lock: threading.Lock
):
    if segment.segment_type == SegmentType.Table:
        with table_engine_lock:
            table_ocr_results = ppstructure_table(
                table_engine, Path(segment_temp_file))
            segment.ocr = table_ocr_results.results
            segment.html = table_ocr_results.html
    elif segment.segment_type == SegmentType.Picture:
        with ocr_lock:
            ocr_results = ppocr(ocr, Path(segment_temp_file))
            segment.ocr = ocr_results.results
    else:
        with ocr_lock:
            ocr_results = ppocr(ocr, Path(segment_temp_file))
        segment.ocr = ocr_results.results


def process_segment(
    segment: Segment,
    page_image_file_paths: dict[int, Path],
    segment_image_density: int,
    segment_image_extension: str,
    segment_image_quality: int,
    segment_image_resize: str,
    ocr_strategy: str,
    ocr: PaddleOCR,
    table_engine: PPStructure,
    ocr_lock: threading.Lock,
    table_engine_lock: threading.Lock
) -> Segment:
    try:
        ocr_needed = ocr_strategy == "All" or (
            ocr_strategy != "Off" and (
                segment.segment_type in [SegmentType.Table, SegmentType.Picture] or
                (ocr_strategy == "Auto" and not segment.text_layer)
            )
        )

        if ocr_needed:
            segment.image = crop_image(
                page_image_file_paths[segment.page_number],
                segment.bbox,
                segment_image_density,
                segment_image_extension,
                segment_image_quality,
                segment_image_resize
            )
            segment_temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{segment_image_extension}", delete=False)
            segment_temp_file.write(base64.b64decode(segment.image))
            segment_temp_file.close()
            try:
                process_segment_ocr(
                    segment,
                    segment_temp_file.name,
                    ocr,
                    table_engine,
                    ocr_lock,
                    table_engine_lock
                )
            finally:
                os.unlink(segment_temp_file.name)
    except Exception as e:
        print(
            f"Error processing segment {segment.segment_type} on page {segment.page_number}: {e}")
    finally:
        segment.create_ocr_text()
        segment.create_html()
        segment.create_markdown()
    return segment


