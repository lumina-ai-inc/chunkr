import asyncio
from collections import deque
from contextlib import asynccontextmanager
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import dotenv
from fastapi import FastAPI, File, UploadFile
import os
from pydantic import BaseModel
import time
import torch
from typing import Dict, List

dotenv.load_dotenv(override=True)

batch_wait_time = float(os.getenv('BATCH_WAIT_TIME', 0.5))
max_batch_size = int(os.getenv('MAX_BATCH_SIZE', 240))

app = FastAPI()

predictor = ocr_predictor('fast_base', 'crnn_vgg16_bn', pretrained=True, 
                         export_as_straight_boxes=True)
if torch.cuda.is_available():
    predictor = predictor.cuda()

pending_tasks = deque()
processing_lock = asyncio.Lock()
batch_event = asyncio.Event()

class OCRTask(BaseModel):
    image_data: bytes
    page_number: int = 0
    future: asyncio.Future = None

    class Config:
        arbitrary_types_allowed = True

class OCRResponse(BaseModel):
    page_content: Dict
    processing_time: float

async def process_ocr_batch(tasks: List[OCRTask]) -> List[OCRResponse]:
    print(f"Number of tasks: {len(tasks)}")
    image_bytes_list = [task.image_data for task in tasks]
    doc = DocumentFile.from_images(image_bytes_list)
    start_time = time.time()
    result = predictor(doc)
    json_output = result.export()
    processing_time = time.time() - start_time
    
    responses = []
    for i in range(len(tasks)):
        page_content = json_output['pages'][i]
        responses.append(OCRResponse(
            page_content=page_content,
            processing_time=processing_time
        ))
    
    # Clean up VRAM
    del doc
    del result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    return responses

async def batch_processor():
    while True:
        try:
            await batch_event.wait()
            
            async with processing_lock:
                if not pending_tasks:
                    batch_event.clear()
                    continue
                
                # Take only max_batch_size tasks
                current_batch = []
                for _ in range(max_batch_size):
                    if not pending_tasks:
                        break
                    current_batch.append(pending_tasks.popleft())
            
                try:
                    # Process single chunk
                    chunk_results = await process_ocr_batch(current_batch)
                    
                    # Set results immediately
                    for task, result in zip(current_batch, chunk_results):
                        if not task.future.done():
                            task.future.set_result(result)
                    
                    # Clear memory
                    del chunk_results
                    del current_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    for task in current_batch:
                        if not task.future.done():
                            task.future.set_exception(e)
            
        except Exception as e:
            print(f"Error in batch processor: {e}")
            batch_event.clear()
            await asyncio.sleep(0.1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    batch_processor_task = asyncio.create_task(batch_processor())
    yield
    batch_processor_task.cancel()
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/ocr", response_model=OCRResponse)
async def create_ocr_task(file: UploadFile = File(...), page_number: int = 0):
    image_data = await file.read()
    
    task = OCRTask(image_data=image_data, page_number=page_number)
    future = asyncio.Future()
    task.future = future
    
    pending_tasks.append(task)
    batch_event.set()
    
    result = await future
    return result

@app.post("/batch")
async def batch_ocr(files: List[UploadFile] = File(...)):
    # Create tasks with futures
    tasks = []
    for file in files:
        image_data = await file.read()
        future = asyncio.Future()
        task = OCRTask(image_data=image_data, future=future)
        tasks.append(task)

    # Process batch
    results = []
    for i in range(0, len(tasks), max_batch_size):
        chunk = tasks[i:i+max_batch_size]
        chunk_results = await process_ocr_batch(chunk)
        results.extend(chunk_results)

    return results

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to") 
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)