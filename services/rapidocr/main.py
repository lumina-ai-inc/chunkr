import asyncio
import json
import numpy as np
from robyn import Robyn, Request
from rapidocr_paddle import RapidOCR
from tempfile import NamedTemporaryFile
import time
import torch
import cv2
import os
import psutil
import gc
app = Robyn(__file__)

# Create an asyncio lock
ocr_lock = asyncio.Lock()

if torch.cuda.is_available():
    print("CUDA is available. Using GPU for RapidOCR.")
    engine = RapidOCR(det_use_cuda=True, rec_use_cuda=True, cls_use_cuda=True)
else:
    print("CUDA is not available. Using CPU for RapidOCR.")
    engine = RapidOCR(det_use_cuda=False, rec_use_cuda=False, cls_use_cuda=False)

# Add this function to check open files
def check_open_files():
    process = psutil.Process()
    open_files = process.open_files()
    # print(f"Number of open files: {len(open_files)}")
    return len(open_files)

@app.post("/ocr")
async def perform_ocr(request: Request):
    request_received_time = time.time()
    
    # Check open files before processing
    check_open_files()
    
    loop = asyncio.get_event_loop()
    process_start_time = time.time()
    
    async with ocr_lock:
        result = await loop.run_in_executor(None, process_ocr, request.files)
    
    process_end_time = time.time()
    
    # Check open files after processing
    check_open_files()
    
    # Force garbage collection
    gc.collect()
    
    request_to_process_delay = process_start_time - request_received_time
    total_processing_time = process_end_time - request_received_time
    print(f"Request to process delay: {request_to_process_delay:.2f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    return {
        "result": result,
        "request_to_process_delay": request_to_process_delay,
        "total_processing_time": total_processing_time
    }

def process_ocr(files):
    temp_file = None
    try:
        temp_file = NamedTemporaryFile(delete=False)
        file_content = next(iter(files.values()))
        temp_file.write(file_content)
        temp_file.flush()
        temp_file_path = temp_file.name
        temp_file.close()  # Explicitly close the file

        start_time = time.time()
        result, _ = engine(temp_file_path)
        end_time = time.time()
    
        serializable_result = json.loads(json.dumps(result, default=lambda x: x.item() if isinstance(x, np.generic) else x))
        processing_time = end_time - start_time
        print(f"OCR processing completed in {processing_time:.2f} seconds")
        return serializable_result
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        raise
    finally:
        if temp_file:
            temp_file.close()  # Ensure the file is closed
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
                print(f"Temporary file {temp_file_path} removed")
            except Exception as e:
                print(f"Error removing temporary file: {e}")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app.start(host="0.0.0.0", port=port)
# img_path = 'test/Example-of-a-complex-table-structure-modified-from-16-2.png'
