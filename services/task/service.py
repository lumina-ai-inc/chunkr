from __future__ import annotations
import bentoml
from pathlib import Path
from typing import Dict, List
from pydantic import Field
from paddleocr import PaddleOCR, PPStructure

from src.ocr import perform_paddle_ocr, ppstructure_table
from src.utils import check_imagemagick_installed
from src.converters import convert_to_img, crop_image
from src.models.ocr_model import OCRResponse


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
        bbox: Dict[str, int]
    ) -> Path:
        height = bbox.get('height', 0)
        left = bbox.get('left', 0)
        top = bbox.get('top', 0)
        width = bbox.get('width', 0)
        return crop_image(file, left, top, left + width, top + height)


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
            recovery=True, return_ocr_result_in_table=True, type="structure")

    @bentoml.api
    def paddle_raw(self, file: Path) -> list:
        return self.ocr.ocr(str(file))

    @bentoml.api
    def paddle(self, file: Path) -> OCRResponse:
        return perform_paddle_ocr(self.ocr, file)

    @bentoml.api
    def paddle_table(self, file: Path) -> list:
        return ppstructure_table(self.table_engine, file)


@bentoml.service(
    name="task",
    resources={"gpu": 1, "cpu": "4"},
    traffic={"timeout": 60}
)
class Task:
    def __init__(self) -> None:
        self.image_service = bentoml.depends(Image)
        self.ocr_service = bentoml.depends(OCR)

    @bentoml.api
    def process(self, file: Path) -> list:
        pass
