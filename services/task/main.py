import aiofiles
import asyncio
import base64
import colorlog
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import gc
import json
import logging
from multiprocessing import cpu_count
import os
from pathlib import Path
import tempfile
import time
from starlette.background import BackgroundTask
import uvicorn

from src.configs.textract_config import login_aws
from src.converters import convert_to_img, convert_to_pdf
from src.models.segment_model import Segment
from src.process import adjust_segments, process_segment

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s%(reset)s:     %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Setting up task service")
    login_aws()
    yield
    logger.info("Shutting down")

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/to_pdf")
async def to_pdf(file: UploadFile = File(...)):
    pdf_path = None
    file_extension = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        content = await file.read()
        temp_file.write(content)
        file_path = Path(temp_file.name)
    try:
        pdf_path = convert_to_pdf(file_path)

        def cleanup_files():
            if file_path.exists():
                os.unlink(file_path)
            if pdf_path and pdf_path.exists():
                os.unlink(pdf_path)

        return FileResponse(
            path=str(pdf_path),
            filename=pdf_path.name,
            background=BackgroundTask(cleanup_files)
        )
    except Exception as e:
        logger.error(f"Error converting file to PDF: {str(e)}")
        if file_path.exists():
            os.unlink(file_path)
        raise HTTPException(
            status_code=500, detail="Failed to convert file to PDF")
    finally:
        gc.collect()  


@app.post("/process")
async def process(

    file: UploadFile = File(...),
    segments: str = Form(...),
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
    # ocr_strategy = "Off"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        file_path = Path(temp_file.name)

    try:
        start_time = time.time()

        segments = json.loads(segments)
        segment_objects = [Segment(**segment) for segment in segments]
        adjust_segments(segment_objects, segment_bbox_offset,
                        page_image_density, pdla_density)

        processed_segments = await process_segments(
            file_path, segment_objects, user_id, task_id, image_folder_location,
            page_image_density, page_image_extension, segment_image_extension,
            segment_image_quality, segment_image_resize, num_workers, ocr_strategy
        )

        logger.debug(f"Total task time: {time.time() - start_time}")
        return processed_segments
    finally:
        os.unlink(file_path)
        gc.collect()

async def process_segments(
    file: Path, segments: list[Segment], user_id: str, task_id: str, image_folder_location: str,
    page_image_density: int, page_image_extension: str, segment_image_extension: str,
    segment_image_quality: int, segment_image_resize: str, num_workers: int, ocr_strategy: str
):
    page_images = await convert_to_img(
        file, page_image_density, page_image_extension)
    page_image_file_paths = {}

    async def write_temp_file(page_number, page_image):
        async with aiofiles.tempfile.NamedTemporaryFile(suffix=f".{page_image_extension}", delete=False) as temp_file:
            await temp_file.write(base64.b64decode(page_image))
            page_image_file_paths[page_number] = Path(temp_file.name)

    await asyncio.gather(*[write_temp_file(page_number, page_image) for page_number, page_image in page_images.items()])

    try:
        processed_segments_dict = {}
        num_workers = num_workers or len(segments) if len(
            segments) > 0 else cpu_count()
        logger.info(f"Number of workers: {num_workers}")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    executor,
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
            ]

            for future in asyncio.as_completed(futures):
                result = await future
                processed_segments_dict[result.segment_id] = result

        return [processed_segments_dict[segment.segment_id] for segment in segments]
    finally:
        await asyncio.gather(*[asyncio.to_thread(os.unlink, path) for path in page_image_file_paths.values()])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8070)
