import base64
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, Form
from multiprocessing import cpu_count
import os
from pathlib import Path
import tempfile
import time
import tqdm
import uvicorn

from src.converters import convert_to_img, convert_to_pdf
from src.models.segment_model import Segment
from src.process import adjust_segments, process_segment

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/to_pdf")
async def to_pdf(file: UploadFile = File(...)):
    return await convert_to_pdf(file)


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    segments: list[Segment] = Form(...),
    user_id: str = Form(...),
    task_id: str = Form(...),
    image_folder_location: str = Form(...),
    page_image_density: int = Form(300),
    page_image_extension: str = Form("png"),
    segment_image_extension: str = Form("jpg"),
    segment_bbox_offset: float = Form(1.5),
    segment_image_quality: int = Form(100),
    segment_image_resize: str = Form(None),
    pdla_density: int = Form(72),
    num_workers: int = Form(None),
    ocr_strategy: str = Form("Auto")
):
    print(f"Processing file: {file.filename}")
    start_time = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        file_path = Path(temp_file.name)

    print(f"File path: {file_path}")
    try:
        adjust_segments(segments, segment_bbox_offset,
                        page_image_density, pdla_density)

        if ocr_strategy == "Off":
            processed_segments = await process_segments_without_ocr(
                segments, num_workers)
        else:
            processed_segments = await process_segments_with_ocr(
                file_path, segments, user_id, task_id, image_folder_location,
                page_image_density, page_image_extension, segment_image_extension,
                segment_image_quality, segment_image_resize, num_workers, ocr_strategy
            )

        print(f"Total task time: {time.time() - start_time}")
        return processed_segments
    finally:
        os.unlink(file_path)


def process_segments_without_ocr(segments: list[Segment], num_workers: int):
    processed_segments_dict = {}
    num_workers = num_workers or len(segments) if len(
        segments) > 0 else cpu_count()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {segment.segment_id: executor.submit(
            lambda s: s.finalize(), segment) for segment in segments}

        for segment_id, future in tqdm.tqdm(futures.items(), desc="Processing segments without OCR", total=len(futures)):
            processed_segments_dict[segment_id] = future.result()

    return [processed_segments_dict[segment.segment_id] for segment in segments]


async def process_segments_with_ocr(
    file: UploadFile, segments: list[Segment], user_id: str, task_id: str, image_folder_location: str,
    page_image_density: int, page_image_extension: str, segment_image_extension: str,
    segment_image_quality: int, segment_image_resize: str, num_workers: int, ocr_strategy: str
):
    page_images = await convert_to_img(
        file, page_image_density, page_image_extension)
    page_image_file_paths = {}

    for page_number, page_image in page_images.items():
        with tempfile.NamedTemporaryFile(suffix=f".{page_image_extension}", delete=False) as temp_file:
            temp_file.write(base64.b64decode(page_image))
            page_image_file_paths[page_number] = Path(temp_file.name)

    try:
        processed_segments_dict = {}
        num_workers = num_workers or len(segments) if len(
            segments) > 0 else cpu_count()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                segment.segment_id: executor.submit(
                    process_segment,
                    user_id,
                    task_id,
                    segment,
                    image_folder_location,
                    page_image_file_paths,
                    segment_image_extension,
                    segment_image_quality,
                    segment_image_resize,
                    ocr_strategy
                )
                for segment in segments
            }

            for segment_id, future in tqdm.tqdm(futures.items(), desc="Processing segments", total=len(futures)):
                processed_segments_dict[segment_id] = future.result()

        return [processed_segments_dict[segment.segment_id] for segment in segments]
    finally:
        for page_image_file_path in page_image_file_paths.values():
            os.unlink(page_image_file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
