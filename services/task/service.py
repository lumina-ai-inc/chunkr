from __future__ import annotations
import bentoml
from pathlib import Path
from typing import Dict
from pydantic import Field
from paddleocr import PaddleOCR

from ocr import perform_paddle_ocr
from utils import check_imagemagick_installed
from converters import convert_to_img


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


@bentoml.service(
    name="ocr",
    resources={"gpu": 1, "cpu": "4"},
    traffic={"timeout": 60}
)
class OCR:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    @bentoml.api 
    def paddle(self, file: Path) -> list:
        return perform_paddle_ocr(self.ocr, file)


@bentoml.service(
    name="segment",
    resources={"gpu": 1},
    traffic={"timeout": 60}
)
class Segment:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    @bentoml.api 
    def paddle(self, file: Path) -> list:
        return perform_paddle_ocr(self.ocr, file)
