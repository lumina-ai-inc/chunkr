import asyncio
import argparse
import cv2  # do not remove
from datetime import datetime
import gc
import json
import numpy as np
import os
from rapidocr_paddle import RapidOCR
from robyn import Robyn, Request
from robyn.logger import Logger
from tempfile import NamedTemporaryFile
import time
import torch

app = Robyn(__file__)
logger = Logger()

ocr_lock = asyncio.Lock()


class LoggingMiddleware:

    def request_info(request: Request):
        ip_address = request.ip_addr
        request_url = request.url.host
        request_path = request.url.path
        request_method = request.method
        request_time = str(datetime.now())

        return {
            "ip_address": ip_address,
            "request_url": request_url,
            "request_path": request_path,
            "request_method": request_method,
            "request_time": request_time,

        }


if torch.cuda.is_available():
    print("CUDA is available. Using GPU for RapidOCR.")
    engine = RapidOCR(det_use_cuda=True, rec_use_cuda=True,
                      cls_use_cuda=True, ocr_order_method="tb-xy")
else:
    print("CUDA is not available. Using CPU for RapidOCR.")
    engine = RapidOCR(det_use_cuda=False,
                      rec_use_cuda=False,
                      cls_use_cuda=False,
                      ocr_order_method="tb-xy")


@app.before_request()
def log_request(request: Request):
    logger.info(f"Received request: %s",
                LoggingMiddleware.request_info(request))
    return request


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def perform_ocr(request: Request):
    loop = asyncio.get_event_loop()
    async with ocr_lock:
        result = await loop.run_in_executor(None, process_ocr, request.files)
    gc.collect()
    return {"result": result}


def process_ocr(files) -> list:
    temp_file = None
    try:
        temp_file = NamedTemporaryFile(delete=False)
        file_content = next(iter(files.values()))
        temp_file.write(file_content)
        temp_file.flush()
        temp_file_path = temp_file.name
        temp_file.close()

        result, _ = engine(temp_file_path)

        serializable_result = [
            [
                [[float(coord) for coord in coords] for coords in item[0]],
                str(item[1]),
                float(item[2])
            ]
            for item in result
        ]

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
