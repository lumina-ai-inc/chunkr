import asyncio
import json
import numpy as np
from robyn import Robyn, Request
from rapidocr_paddle import RapidOCR
from tempfile import NamedTemporaryFile
import time
import torch

app = Robyn(__file__)

if torch.cuda.is_available():
    print("CUDA is available. Using GPU for RapidOCR.")
    engine = RapidOCR(det_use_cuda=True, rec_use_cuda=True)
else:
    print("CUDA is not available. Using CPU for RapidOCR.")
    engine = RapidOCR(det_use_cuda=False, rec_use_cuda=False)


@app.post("/ocr")
async def perform_ocr(request: Request):
    # Measure delay between receiving request and hitting process_ocr
    request_received_time = time.time()
    
    # Move the OCR processing to a separate function
    loop = asyncio.get_event_loop()
    process_start_time = time.time()
    result = await loop.run_in_executor(None, process_ocr, request.files)
    process_end_time = time.time()
    
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
    with NamedTemporaryFile(delete=False) as temp_file:
        file_content = next(iter(files.values()))
        temp_file.write(file_content)
        temp_file.flush()
        start_time = time.time()
        result, _ = engine(temp_file.name)
        end_time = time.time()
    
    # Convert numpy types to Python native types
    serializable_result = json.loads(json.dumps(result, default=lambda x: x.item() if isinstance(x, np.generic) else x))
    processing_time = end_time - start_time
    print(f"OCR processing completed in {processing_time:.2f} seconds")
    return serializable_result

if __name__ == "__main__":
    app.start(host="0.0.0.0", port=8000)
# img_path = 'test/Example-of-a-complex-table-structure-modified-from-16-2.png'
