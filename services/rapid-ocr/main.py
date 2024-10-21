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
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
import requests
import tarfile
import shutil

# Load environment variables
load_dotenv(override=True)

app = Robyn(__file__)
logger = Logger()

def download_and_extract_model(url, model_type):
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    filename = os.path.join(models_dir, f'{model_type}_model.tar')
    
    # Download the file
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Extract the tar file
    with tarfile.open(filename, 'r') as tar:
        tar.extractall(path=models_dir)
    
    # Move the inference.pdmodel file
    extracted_dir = os.path.join(models_dir, f'ch_PP-OCRv4_{model_type}_infer')
    src_file = os.path.join(extracted_dir, 'inference.pdmodel')
    dst_file = os.path.join(models_dir, f'{model_type}_model.pdmodel')
    shutil.move(src_file, dst_file)
    
    # Clean up
    os.remove(filename)
    shutil.rmtree(extracted_dir)

def download_models():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    det_model_path = 'models/det_model.pdmodel'
    rec_model_path = 'models/rec_model.pdmodel'
    
    if not os.path.exists(det_model_path):
        download_and_extract_model('https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar', 'det')
    
    if not os.path.exists(rec_model_path):
        download_and_extract_model('https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar', 'rec')

# Call the download function
download_models()

# Define the number of OCR engines to create
NUM_ENGINES = int(os.getenv('RAPID_OCR__NUM_ENGINES', 4))

# Create a list of OCR engines and corresponding semaphores
engines = []
engine_semaphores = []

for _ in range(NUM_ENGINES):
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU for RapidOCR.")
        engine = RapidOCR(det_use_cuda=True, rec_use_cuda=True,
                          cls_use_cuda=True, ocr_order_method="tb-xy", 
                          rec_model_path="models/rec_model.pdmodel", 
                          det_model_path="models/det_model.pdmodel")
    else:
        print("CUDA is not available. Using CPU for RapidOCR.")
        engine = RapidOCR(det_use_cuda=False,
                          rec_use_cuda=False,
                          cls_use_cuda=False,
                          ocr_order_method="tb-xy",
                          rec_model_path="models/rec_model.pdmodel", 
                          det_model_path="models/det_model.pdmodel")
    engines.append(engine)
    engine_semaphores.append(asyncio.Semaphore(1))


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
    
    # Find an available engine
    for i, semaphore in enumerate(engine_semaphores):
        if semaphore.locked():
            continue
        
        async with semaphore:
            result = await loop.run_in_executor(None, process_ocr, request.files, engines[i])
            gc.collect()
            return {"result": result}
    
    # If all engines are busy, wait for the first available one
    async with engine_semaphores[0]:
        result = await loop.run_in_executor(None, process_ocr, request.files, engines[0])
        gc.collect()
        return {"result": result}


def serialize_ocr_result(result):
    print(result)
    if result is None:
        return []
    return [
        [
            [[float(coord) for coord in coords if coord is not None]
             for coords in item[0] if coords],
            str(item[1]) if item[1] is not None else "",
            float(item[2]) if item[2] is not None else 0.0
        ]
        for item in result if item is not None
    ]


def process_ocr(files, engine) -> list:
    temp_file = None
    try:
        temp_file = NamedTemporaryFile(delete=False)
        #cant send multiple 
        def preprocess_image(image_content):
            # Read the image content
            image = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Denoise the image
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # Encode the preprocessed image back to bytes
            _, preprocessed_image = cv2.imencode('.png', denoised)
            
            return preprocessed_image.tobytes()

        file_content = next(iter(files.values()))
        preprocessed_content = preprocess_image(file_content)
        temp_file.write(preprocessed_content)
        temp_file.flush()
        temp_file_path = temp_file.name
        temp_file.close()

        result, _ = engine(temp_file_path)

        return serialize_ocr_result(result)
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
