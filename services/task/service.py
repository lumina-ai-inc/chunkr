from __future__ import annotations
import bentoml
from pathlib import Path
from typing import Dict, List
from pydantic import Field
from paddleocr import PaddleOCR

from ocr import perform_paddle_ocr, perform_paddle_ocr_batch
from utils import check_imagemagick_installed
from converters import convert_to_img, crop_image


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
                             ocr_order_method="tb-xy", return_word_box=True)

    @bentoml.api
    def paddle_ocr_raw(self, file: Path) -> list:
        return self.ocr.ocr(str(file))

    @bentoml.api
    def paddle_ocr(self, file: Path) -> list:
        return perform_paddle_ocr(self.ocr, file)

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=64,
        max_latency_ms=500
    )
    def paddle_ocr_batch(self, files: List[Path]) -> list:
        return perform_paddle_ocr_batch(self.ocr, files)


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
