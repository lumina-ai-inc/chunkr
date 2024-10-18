from ocr_service import process_table_image
import asyncio

import cv2  # do not remove
from datetime import datetime

import argparse
import gc
from robyn import Robyn, Request
from robyn.logger import Logger


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


@app.get("/readiness")
def readiness():
    return {"status": "ready"}


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/table")
async def process_table(request: Request):
    async with ocr_lock:
        table_data = []
        for file_content in request.files.values():
            result = await process_table_image(file_content)
            table_data.append(result)
    gc.collect()
    return {"result": table_data[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OCR service")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8010,
                        help="Port to run the server on")

    args = parser.parse_args()

    app.start(host=args.host, port=args.port)
