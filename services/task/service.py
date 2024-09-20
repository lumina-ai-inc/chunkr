from __future__ import annotations
import base64
import bentoml
import multiprocessing
import os
from paddleocr import PaddleOCR, PPStructure
from pathlib import Path
from pydantic import Field
import tempfile
from typing import Dict

from src.converters import convert_to_img, crop_image
from src.models.ocr_model import OCRResponse
from src.models.segment_model import Segment, SegmentType
from src.ocr import ppocr, ppocr_raw, ppstructure_table, ppstructure_table_raw
from src.utils import check_imagemagick_installed


@bentoml.service(
    name="image",
    resources={"cpu": "2"},
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
        extension: str = Field(default="png", description="Image extension")
    ) -> str:
        # TODO: Add png support
        return crop_image(str(file), left, top, left + width, top + height, extension)


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
    image_service = bentoml.depends(Image)
    ocr_service = bentoml.depends(OCR)

    @bentoml.api
    def images_from_file(
        self,
        file: Path,
        density: int = Field(default=300, description="Image density in DPI"),
        extension: str = Field(default="png", description="Image extension")
    ) -> Dict[int, str]:
        return self.image_service.convert_to_img(file, density, extension)

    @bentoml.api
    def process(
            self,
            file: Path,
            segments: list[Segment],
            image_density: int = Field(
                default=300, description="Image density in DPI for page images"),
            page_image_extension: str = Field(
                default="png", description="Image extension for page images"),
            segment_image_extension: str = Field(
                default="jpg", description="Image extension for segment images")
    ) -> list[Segment]:
        print("Processing started")
        page_images = self.image_service.convert_to_img(
            file, image_density, page_image_extension)
        page_image_file_paths: dict[int, Path] = {}
        for page_number, page_image in page_images.items():
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{page_image_extension}", delete=False)
            temp_file.write(base64.b64decode(page_image))
            temp_file.close()
            page_image_file_paths[page_number] = Path(temp_file.name)
        try:
            for segment in segments:
                segment.image = self.image_service.crop_image(
                    page_image_file_paths[segment.page_number], segment.left, segment.top, segment.width, segment.height, segment_image_extension)
                segment_temp_file = tempfile.NamedTemporaryFile(
                    suffix=f".{segment_image_extension}", delete=False)
                segment_temp_file.write(base64.b64decode(segment.image))
                segment_temp_file.close()
                try:
                    if segment.segment_type == SegmentType.Table:
                        segment.ocr = self.ocr_service.paddle_table(
                            Path(segment_temp_file.name))
                    else:
                        segment.ocr = self.ocr_service.paddle_ocr(
                            Path(segment_temp_file.name))
                finally:
                    os.unlink(segment_temp_file.name)
        finally:
            for page_image_file_path in page_image_file_paths.values():
                os.unlink(page_image_file_path)
        return segments
