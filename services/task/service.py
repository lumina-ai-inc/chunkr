from __future__ import annotations
import base64
import bentoml
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from paddleocr import PaddleOCR, PPStructure
from pathlib import Path
from pydantic import Field
import tempfile
import tqdm
from typing import Dict, Optional, List
import threading

from src.converters import convert_to_img, crop_image
from src.models.ocr_model import OCRResult, BoundingBox
from src.models.segment_model import BaseSegment, Segment, SegmentType
from src.ocr import ppocr, ppocr_raw, ppstructure_table, ppstructure_table_raw
from src.utils import check_imagemagick_installed, convert_base_segment_to_segment
from src.process import adjust_base_segments


@bentoml.service(
    name="image",
    resources={"cpu": "4"},
    traffic={"timeout": 60}
)
class Image:
    def __init__(self) -> None:
        check_imagemagick_installed()

    @bentoml.api
    def convert_to_img(
        self,
        file: Path,
        density: int = Field(default=300, description="Image density in DPI"),
        extension: str = Field(default="png", description="Image extension")
    ) -> Dict[int, str]:
        return convert_to_img(file, density, extension)

    @bentoml.api
    def crop_image(
        self,
        file: Path,
        bbox: BoundingBox,
        density: int = Field(default=300, description="Image density in DPI"),
        extension: str = Field(default="png", description="Image extension"),
        quality: int = Field(default=100, description="Image quality (0-100)"),
        resize: Optional[str] = Field(
            default=None, description="Image resize dimensions (e.g., '800x600')")
    ) -> str:
        return crop_image(file, bbox, density, extension, quality, resize)


@bentoml.service(
    name="ocr",
    resources={"gpu": 1, "cpu": "4"},
    traffic={"timeout": 60}
)
class OCR:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en",
                             ocr_order_method="tb-xy")
        self.table_engine = PPStructure(
            recovery=True, return_ocr_result_in_table=True, layout=False, structure_version="PP-StructureV2")

    @bentoml.api
    def paddle_ocr_raw(self, file: Path) -> list:
        return ppocr_raw(self.ocr, file)

    @bentoml.api
    def paddle_ocr(self, file: Path) -> List[OCRResult]:
        return ppocr(self.ocr, file)

    @bentoml.api
    def paddle_table_raw(self, file: Path) -> list:
        return ppstructure_table_raw(self.table_engine, file)

    @bentoml.api
    def paddle_table(self, file: Path) -> List[OCRResult]:
        return ppstructure_table(self.table_engine, file)


@bentoml.service(
    name="task",
    resources={"gpu": 1, "cpu": "4"},
    traffic={"timeout": 600}
)
class Task:
    def __init__(self) -> None:
        check_imagemagick_installed()
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en",
                             ocr_order_method="tb-xy", show_logs=False)
        self.table_engine = PPStructure(
            recovery=True, return_ocr_result_in_table=True, layout=False, structure_version="PP-StructureV2")
        self.ocr_lock = threading.Lock()
        self.table_engine_lock = threading.Lock()

    @bentoml.api
    def images_from_file(
        self,
        file: Path,
        density: int = Field(default=300, description="Image density in DPI"),
        extension: str = Field(default="png", description="Image extension")
    ) -> Dict[int, str]:
        return convert_to_img(file, density, extension)

    def process_segment(self, segment: Segment, page_image_file_paths: dict[int, Path], segment_image_density: int, segment_image_extension: str, segment_image_quality: int, segment_image_resize: str) -> Segment:
        try:
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
                if segment.segment_type == SegmentType.Table:
                    with self.table_engine_lock:
                        table_ocr_results = ppstructure_table(
                            self.table_engine, Path(segment_temp_file.name))
                        segment.ocr = table_ocr_results.results
                        segment.html = table_ocr_results.html
                else:
                    with self.ocr_lock:
                        ocr_results = ppocr(
                            self.ocr, Path(segment_temp_file.name))
                        segment.ocr = ocr_results.results
                segment.upsert_html()
                segment.create_markdown()
            finally:
                os.unlink(segment_temp_file.name)
        except Exception as e:
            print(
                f"Error processing segment {segment.segment_type} on page {segment.page_number}: {e}")
        return segment

    @bentoml.api
    def process(
        self,
        file: Path,
        base_segments: list[BaseSegment],
        page_image_density: int = Field(
            default=300, description="Image density in DPI for page images"),
        page_image_extension: str = Field(
            default="png", description="Image extension for page images"),
        segment_image_extension: str = Field(
            default="jpg", description="Image extension for segment images"),
        segment_image_density: int = Field(
            default=300, description="Image density in DPI for segment images"),
        segment_bbox_offset: float = Field(
            default=1.5, description="Offset for segment bbox"),
        segment_image_quality: int = Field(
            default=100, description="Image quality (0-100) for segment images"),
        segment_image_resize: str = Field(
            default=None, description="Image resize dimensions (e.g., '800x600') for segment images"),
        pdla_density: int = Field(
            default=72, description="Image density in DPI for pdla"),
        num_workers: int = Field(
            default=4, description="Number of worker threads for segment processing")
    ) -> list[Segment]:
        print("Processing started")
        adjust_base_segments(base_segments, segment_bbox_offset,
                             page_image_density, pdla_density)
        segments = [convert_base_segment_to_segment(base_segment)
                    for base_segment in base_segments]
        page_images = convert_to_img(
            file, page_image_density, page_image_extension)
        page_image_file_paths: dict[int, Path] = {}
        for page_number, page_image in page_images.items():
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{page_image_extension}", delete=False)
            temp_file.write(base64.b64decode(page_image))
            temp_file.close()
            page_image_file_paths[page_number] = Path(temp_file.name)
        print("Pages converted to images")
        try:
            print("Segment processing started")
            processed_segments = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for segment in segments:
                    future = executor.submit(
                        self.process_segment,
                        segment,
                        page_image_file_paths,
                        segment_image_density,
                        segment_image_extension,
                        segment_image_quality,
                        segment_image_resize
                    )
                    futures.append(future)

                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing segments"):
                    processed_segments.append(future.result())

            print("Segment processing finished")
        finally:
            for page_image_file_path in page_image_file_paths.values():
                os.unlink(page_image_file_path)
        return processed_segments
