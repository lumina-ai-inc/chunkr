from __future__ import annotations
import base64
import bentoml
from concurrent.futures import ThreadPoolExecutor, Future
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
from src.models.segment_model import BaseSegment, Segment
from src.ocr import ppocr, ppocr_raw, ppstructure_table, ppstructure_table_raw
from src.process import adjust_base_segments, process_segment


@bentoml.service(
    name="image",
    resources={"cpu": "4"},
    traffic={"timeout": 60}
)
class Image:
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
        extension: str = Field(default="png", description="Image extension"),
        quality: int = Field(default=100, description="Image quality (0-100)"),
        resize: Optional[str] = Field(
            default=None, description="Image resize dimensions (e.g., '800x600')")
    ) -> str:
        return crop_image(file, bbox, extension, quality, resize)


@bentoml.service(
    name="ocr",
    resources={"gpu": 1, "cpu": "4"},
    traffic={"timeout": 60}
)
class OCR:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en",
                             ocr_order_method="tb-xy", show_log=False)
        self.table_engine = PPStructure(
            recovery=True, return_ocr_result_in_table=True, layout=False, structure_version="PP-StructureV2", show_log=False)

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
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en",
                             ocr_order_method="tb-xy", show_log=False)
        # todo: add lang support
        self.table_engine = PPStructure(
            recovery=True, return_ocr_result_in_table=True, layout=False, structure_version="PP-StructureV2", show_log=False)
        self.ocr_lock: threading.Lock = threading.Lock()
        self.table_engine_lock: threading.Lock = threading.Lock()

    @bentoml.api
    def images_from_file(
        self,
        file: Path,
        density: int = Field(default=300, description="Image density in DPI"),
        extension: str = Field(default="png", description="Image extension")
    ) -> Dict[int, str]:
        return convert_to_img(file, density, extension)

    @bentoml.api
    def process(
        self,
        file: Path,
        base_segments: list[BaseSegment],
        image_folder_location: str = Field(
            description="S3 path for page images"),
        page_image_density: int = Field(
            default=300, description="Image density in DPI for page images"),
        page_image_extension: str = Field(
            default="png", description="Image extension for page images"),
        segment_image_extension: str = Field(
            default="jpg", description="Image extension for segment images"),
        segment_bbox_offset: float = Field(
            default=1.5, description="Offset for segment bbox"),
        segment_image_quality: int = Field(
            default=100, description="Image quality (0-100) for segment images"),
        segment_image_resize: str = Field(
            default=None, description="Image resize dimensions (e.g., '800x600') for segment images"),
        pdla_density: int = Field(
            default=72, description="Image density in DPI for pdla"),
        num_workers: int = Field(
            default=None, description="Number of worker threads for segment processing"),
        ocr_strategy: str = Field(
            default="Auto", description="OCR strategy: 'Auto', 'All', or 'Off'")
    ) -> list[Segment]:
        print("Processing started")
        adjust_base_segments(base_segments, segment_bbox_offset,
                             page_image_density, pdla_density)
        segments = [Segment.from_base_segment(base_segment)
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
            processed_segments_dict = {}
            with ThreadPoolExecutor(max_workers=num_workers or len(segments)) as executor:
                futures: dict[str, Future] = {}
                for segment in segments:
                    future = executor.submit(
                        process_segment,
                        segment,
                        image_folder_location,
                        page_image_file_paths,
                        segment_image_extension,
                        segment_image_quality,
                        segment_image_resize,
                        ocr_strategy,
                        self.ocr,
                        self.table_engine,
                        self.ocr_lock,
                        self.table_engine_lock
                    )
                    futures[segment.segment_id] = future

                total_segments = len(futures)
                for segment_id, future in tqdm.tqdm(futures.items(), desc="Processing segments", total=total_segments):
                    processed_segments_dict[segment_id] = future.result()

            processed_segments = [
                processed_segments_dict[base_segment.segment_id] for base_segment in base_segments]
            print("Segment processing finished")
        finally:
            for page_image_file_path in page_image_file_paths.values():
                os.unlink(page_image_file_path)
        return processed_segments
