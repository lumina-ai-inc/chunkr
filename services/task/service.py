from __future__ import annotations
import base64
import bentoml
from multiprocessing import Pool, cpu_count
import os
from paddleocr import PaddleOCR, PPStructure
from pathlib import Path
from pydantic import Field
import tempfile
from typing import Dict, Optional

from src.converters import convert_to_img, crop_image
from src.models.ocr_model import OCRResponse
from src.models.segment_model import Segment, SegmentType
from src.ocr import ppocr, ppocr_raw, ppstructure_table, ppstructure_table_raw
from src.utils import check_imagemagick_installed, ImprovedSpeller
from src.process import adjust_segments


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
        left: float,
        top: float,
        width: float,
        height: float,
        density: int = Field(default=300, description="Image density in DPI"),
        extension: str = Field(default="png", description="Image extension"),
        quality: int = Field(default=100, description="Image quality (0-100)"),
        resize: Optional[str] = Field(
            default=None, description="Image resize dimensions (e.g., '800x600')")
    ) -> str:
        return crop_image(file, left, top, left + width, top + height, density, extension, quality, resize)


@bentoml.service(
    name="ocr",
    resources={"gpu": 1, "cpu": "4"},
    traffic={"timeout": 60}
)
class OCR:
    def __init__(self) -> None:
        self.spell = ImprovedSpeller(only_replacements=True)
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en",
                             ocr_order_method="tb-xy")
        self.table_engine = PPStructure(
            recovery=True, return_ocr_result_in_table=True, layout=False, structure_version="PP-StructureV2")

    @bentoml.api
    def paddle_ocr_raw(self, file: Path) -> list:
        return ppocr_raw(self.ocr, file)

    @bentoml.api
    def paddle_ocr(self, file: Path) -> OCRResponse:
        return ppocr(self.ocr, file)

    @bentoml.api
    def paddle_table_raw(self, file: Path) -> list:
        return ppstructure_table_raw(self.table_engine, file)

    @bentoml.api
    def paddle_table(self, file: Path) -> OCRResponse:
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
                             ocr_order_method="tb-xy")
        self.table_engine = PPStructure(
            recovery=True, return_ocr_result_in_table=True, layout=False, structure_version="PP-StructureV2")

    @bentoml.api
    def images_from_file(
        self,
        file: Path,
        density: int = Field(default=300, description="Image density in DPI"),
        extension: str = Field(default="png", description="Image extension")
    ) -> Dict[int, str]:
        return convert_to_img(file, density, extension)

    def process_segment(self, segment, page_image_file_paths, segment_image_density, segment_image_extension, segment_image_quality, segment_image_resize):
        try:
            segment.image = crop_image(
                page_image_file_paths[segment.page_number],
                segment.left,
                segment.top,
                segment.width,
                segment.height,
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
                    segment.ocr = ppstructure_table(
                        self.table_engine, Path(segment_temp_file.name))
                else:
                    segment.ocr = ppocr(
                        self.ocr, Path(segment_temp_file.name))
                    segment.update_text_ocr()
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
            segments: list[Segment],
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
            num_processes: int = Field(
                default=cpu_count(), description="Number of processes to use for segment processing")
    ) -> list[Segment]:
        print("Processing started")
        adjust_segments(segments, segment_bbox_offset,
                        page_image_density, pdla_density)
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
            with Pool(processes=num_processes) as pool:
                processed_segments = pool.starmap(
                    self.process_segment,
                    [(segment, page_image_file_paths, segment_image_density,
                      segment_image_extension, segment_image_quality, segment_image_resize)
                     for segment in segments]
                )
            print("Segment processing finished")
        finally:
            for page_image_file_path in page_image_file_paths.values():
                os.unlink(page_image_file_path)
        return processed_segments
