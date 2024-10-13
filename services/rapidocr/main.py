import asyncio
import argparse
import cv2  # do not remove
import gc
import json
import numpy as np
import os
from rapidocr_paddle import RapidOCR
from robyn import Robyn, Request, Response, Logger
from tempfile import NamedTemporaryFile
import time
import torch

app = Robyn(__file__)
logger = Logger(app)

ocr_lock = asyncio.Lock()

if torch.cuda.is_available():
    print("CUDA is available. Using GPU for RapidOCR.")
    engine = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
else:
    print("CUDA is not available. Using CPU for RapidOCR.")
    engine = RapidOCR(det_use_cuda=False,
                      rec_use_cuda=False, cls_use_cuda=False)


@app.before_request()
async def log_request(request: Request):
    logger.info(f"Received request: %s", request)


@app.after_request()
async def log_response(response: Response):
    logger.info(f"Sending response: %s", response)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def perform_ocr(request: Request):
    loop = asyncio.get_event_loop()
    async with ocr_lock:
        result = await loop.run_in_executor(None, process_ocr, request.files)
    gc.collect()
    return result


def process_ocr(files):
    temp_file = None
    try:
        temp_file = NamedTemporaryFile(delete=False)
        file_content = next(iter(files.values()))
        temp_file.write(file_content)
        temp_file.flush()
        temp_file_path = temp_file.name
        temp_file.close()

        result, _ = engine(temp_file_path)

        serializable_result = json.loads(json.dumps(
            result, default=lambda x: x.item() if isinstance(x, np.generic) else x))
        return serializable_result
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        raise
    finally:
        if temp_file:
            temp_file.close()
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error removing temporary file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OCR service")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")

    args = parser.parse_args()

    app.start(host=args.host, port=args.port)
