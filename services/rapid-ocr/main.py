import asyncio
import argparse
import cv2
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
from PIL import Image
import requests
import tarfile
import shutil
import atexit

# Load environment variables
load_dotenv(override=True)

app = Robyn(__file__)

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
        download_and_extract_model('https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/en_PP-OCRv4_det_infer.tar', 'det')
    
    if not os.path.exists(rec_model_path):
        download_and_extract_model('https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/en_PP-OCRv4_rec_infer.tar', 'rec')

    if not os.path.exists(det_model_path):
        raise FileNotFoundError(f"{det_model_path} does not exist after download and extraction.")
    if not os.path.exists(rec_model_path):
        raise FileNotFoundError(f"{rec_model_path} does not exist after download and extraction.")

# Call the download function
download_models()

# Define the number of OCR engines to create
NUM_ENGINES = int(os.getenv('RAPID_OCR__NUM_ENGINES', 4))

print(f"Initializing {NUM_ENGINES} OCR engines.")

# Initialize OCR engines
ocr_engines = []
if torch.cuda.is_available():
    print(f"CUDA is available. Initializing GPU OCR engines.")
    for i in range(NUM_ENGINES):
        print(f"Initializing GPU OCR engine {i + 1}.")
        engine = RapidOCR(
            det_use_cuda=True,
            rec_use_cuda=True,
            cls_use_cuda=True,
            ocr_order_method="tb-xy",
            rec_model_dir="models",
            det_model_dir="models"
        )
        ocr_engines.append(engine)
else:
    print(f"CUDA is not available. Initializing CPU OCR engines.")
    for i in range(NUM_ENGINES):
        print(f"Initializing CPU OCR engine {i + 1}.")
        engine = RapidOCR(
            det_use_cuda=False,
            rec_use_cuda=False,
            cls_use_cuda=False,
            ocr_order_method="tb-xy",
            rec_model_dir="models",
            det_model_dir="models"
        )
        ocr_engines.append(engine)

# Semaphore to limit concurrent OCR tasks
ocr_semaphore = asyncio.Semaphore(NUM_ENGINES)

# Counter for round-robin engine selection
engine_counter = -1
# engine_lock = asyncio.Lock()

def serialize_ocr_result(result):
    if result is None:
        return []
    return [
        {
            "coordinates": [[float(coord) for coord in coords if coord is not None] for coords in item[0] if coords],
            "text": str(item[1]) if item[1] is not None else "",
            "confidence": float(item[2]) if item[2] is not None else 0.0
        }
        for item in result if item is not None
    ]

def preprocess_image(image_content):
    image = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    _, preprocessed_image = cv2.imencode('.png', denoised)
    return preprocessed_image.tobytes()

async def run_ocr(file_content, engine):
    loop = asyncio.get_event_loop()
    temp_file_path = None
    try:
        # Preprocess image if needed
        # file_content = preprocess_image(file_content)
        
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        # Run OCR in executor to avoid blocking
        result, _ = await loop.run_in_executor(None, engine, temp_file_path)
        return serialize_ocr_result(result)
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return {"error": str(e)}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Error removing temporary file {temp_file_path}: {e}")

def get_next_engine_index():
    global engine_counter
    engine_counter = (engine_counter + 1) % NUM_ENGINES
    return engine_counter

# Register shutdown handler
def shutdown():
    print("Shutting down OCR engines...")
    # If RapidOCR has a shutdown method, call it here
    print("OCR engines have been shut down.")

atexit.register(shutdown)

class LoggingMiddleware:
    @staticmethod
    def request_info(request: Request):
        return {
            "ip_address": request.ip_addr,
            "request_url": request.url.host,
            "request_path": request.url.path,
            "request_method": request.method,
            "request_time": str(datetime.now()),
        }

@app.before_request()
async def log_request(request: Request):
    print(f"Received request: {LoggingMiddleware.request_info(request)}")
    return request

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/ocr")
async def perform_ocr(request: Request):

    try:
        files = request.files
        if not files:
            print("No files provided in the request.")
            return {"error": "No files provided."}

        file_key, file = next(iter(files.items()))
        file_content = file  # Directly use the bytes object

        async with ocr_semaphore:
            print(f"Start processing request at {datetime.now()}")
            start_time = time.time()

            # Get the next engine index using round-robin
            engine_index = get_next_engine_index()
            engine = ocr_engines[engine_index]
            
            # Process the OCR request using the selected OCR engine
            result = await run_ocr(file_content, engine)
        
        gc.collect()
        end_time = time.time()
        print(f"Finished processing request at {datetime.now()}, Time taken: {end_time - start_time:.2f} seconds")
        return {"result": result}

    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the OCR service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8020, help="Port to run the server on")
    args = parser.parse_args()

    try:
        app.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("Server shutdown initiated by user.")