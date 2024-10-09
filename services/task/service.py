from __future__ import annotations
import base64
import bentoml
from concurrent.futures import ThreadPoolExecutor, Future
from multiprocessing import cpu_count
import os
from paddleocr import PaddleOCR, PPStructure
from pathlib import Path
from pydantic import Field
import tempfile
import tqdm

from src.converters import convert_to_img, convert_to_pdf
from src.models.segment_model import Segment
from src.process import adjust_segments, process_segment


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

    @bentoml.api
    def to_pdf(self, file: Path) -> Path:
        return convert_to_pdf(file)

    @bentoml.api
    def process(
        self,
        file: Path,
        segments: list[Segment],
        user_id: str = Field(
            description="User ID"),
        task_id: str = Field(
            description="Task ID"),
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
        page_images = []
        adjust_segments(segments, segment_bbox_offset,
                        page_image_density, pdla_density)
        processed_segments = []
        num_workers = num_workers or len(segments) if len(
            segments) > 0 else cpu_count()
        print(num_workers)
        if ocr_strategy == "Off":
            processed_segments_dict = {}

            def finalize_segment(segment: Segment):
                segment.finalize()
                return segment
            with ThreadPoolExecutor(max_workers=num_workers or len(segments)) as executor:
                futures: dict[str, Future] = {}
                for segment in segments:
                    future = executor.submit(
                        finalize_segment,
                        segment
                    )
                    futures[segment.segment_id] = future

                total_segments = len(futures)
                for segment_id, future in tqdm.tqdm(futures.items(), desc="Processing segments", total=total_segments):
                    processed_segments_dict[segment_id] = future.result()
            processed_segments = [
                processed_segments_dict[segment.segment_id] for segment in segments]
        else:
            page_images = convert_to_img(
                file, page_image_density, page_image_extension)
            page_image_file_paths: dict[int, Path] = {}
            for page_number, page_image in page_images.items():
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=f".{page_image_extension}", delete=False)
                temp_file.write(base64.b64decode(page_image))
                temp_file.close()
                page_image_file_paths[page_number] = Path(temp_file.name)
            try:
                processed_segments_dict = {}

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures: dict[str, Future] = {}
                    for segment in segments:
                        future = executor.submit(
                            process_segment,
                            user_id,
                            task_id,
                            segment,
                            image_folder_location,
                            page_image_file_paths,
                            segment_image_extension,
                            segment_image_quality,
                            segment_image_resize,
                            ocr_strategy,
                            self.ocr,
                            self.table_engine
                        )
                        futures[segment.segment_id] = future

                    total_segments = len(futures)
                    for segment_id, future in tqdm.tqdm(futures.items(), desc="Processing segments", total=total_segments):
                        processed_segments_dict[segment_id] = future.result()

                processed_segments = [
                    processed_segments_dict[segment.segment_id] for segment in segments]
            finally:
                for page_image_file_path in page_image_file_paths.values():
                    os.unlink(page_image_file_path)
        return processed_segments
