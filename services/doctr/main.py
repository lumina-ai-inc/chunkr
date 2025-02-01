import asyncio
from collections import deque
from contextlib import asynccontextmanager
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import dotenv
from fastapi import FastAPI, File, UploadFile
import numpy as np
import os
from pydantic import BaseModel
from pydantic.generics import GenericModel
import time
import torch
from typing import Dict, List, TypeVar, Generic, Optional

dotenv.load_dotenv(override=True)

batch_wait_time = float(os.getenv('OCR_BATCH_WAIT_TIME', 0.5))
max_batch_size = int(os.getenv('OCR_MAX_BATCH_SIZE', 100))

predictor = ocr_predictor('fast_base', 'master', pretrained=True, 
                         export_as_straight_boxes=True)
if torch.cuda.is_available():
    print("Using GPU")
    predictor = predictor.cuda().half()
else:
    print("Using CPU")

os.environ['USE_TORCH'] = 'YES'
os.environ['USE_TF'] = 'NO'
pending_tasks = deque()
processing_lock = asyncio.Lock()
batch_event = asyncio.Event()

class OCRTask(BaseModel):
    image_data: bytes
    page_number: int = 0
    future: asyncio.Future = None

    class Config:
        arbitrary_types_allowed = True

T = TypeVar("T")
class Detection(GenericModel, Generic[T]):
    value: Optional[T]
    confidence: Optional[float]

class Word(BaseModel):
    value: str
    confidence: float
    geometry: List[List[float]]
    objectness_score: float
    crop_orientation: Detection[int]

class Line(BaseModel):
    geometry: List[List[float]]
    objectness_score: float
    words: List[Word]

class Block(BaseModel):
    geometry: List[List[float]]
    objectness_score: float
    lines: List[Line]
    artefacts: List[str]

class PageContent(BaseModel):
    page_idx: int
    dimensions: List[int]
    orientation: Detection[float]
    language: Detection[str]
    blocks: List[Block]

class OCRResponse(BaseModel):
    page_content: PageContent
    processing_time: float

async def process_ocr_batch(tasks: List[OCRTask]) -> List[OCRResponse]:
    print(f"Number of tasks: {len(tasks)}")
    
    image_bytes_list = [task.image_data for task in tasks]
    doc = DocumentFile.from_images(image_bytes_list)
    
    start_time = time.time()
    result = predictor(doc)
    responses = []
    
    for page_idx, page in enumerate(result.pages):
        blocks = []
        for block in page.blocks:
            lines = []
            for line in block.lines:
                words = []
                for word in line.words:
                    words.append(Word(
                        value=word.value,
                        confidence=word.confidence,
                        geometry=word.geometry,
                        objectness_score=1.0,
                        crop_orientation=Detection(value=0, confidence=1.0)
                    ))
                lines.append(Line(
                    geometry=line.geometry,
                    objectness_score=1.0,
                    words=words
                ))
            blocks.append(Block(
                geometry=block.geometry,
                objectness_score=1.0,
                lines=lines,
                artefacts=[]
            ))
            
        page_content = PageContent(
            page_idx=page_idx,
            dimensions=page.dimensions,
            orientation=Detection(value=0.0, confidence=1.0),
            language=Detection(value="en", confidence=1.0),
            blocks=blocks
        )
        
        responses.append(OCRResponse(
            page_content=page_content,
            processing_time=time.time() - start_time
        ))
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return responses

async def batch_processor():
    while True:
        try:
            await batch_event.wait()
            
            async with processing_lock:
                if not pending_tasks:
                    batch_event.clear()
                    continue
                
                current_batch = []
                for _ in range(max_batch_size):
                    if not pending_tasks:
                        break
                    current_batch.append(pending_tasks.popleft())
            
                try:
                    chunk_results = await process_ocr_batch(current_batch)
                    
                    for task, result in zip(current_batch, chunk_results):
                        if not task.future.done():
                            task.future.set_result(result)
                    
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
    total_start_time = time.time()
    tasks = []
    for file in files:
        image_data = await file.read()
        future = asyncio.Future()
        task = OCRTask(image_data=image_data, future=future)
        tasks.append(task)

    results = []
    for i in range(0, len(tasks), max_batch_size):
        chunk = tasks[i:i+max_batch_size]
        chunk_start_time = time.time()
        chunk_results = await process_ocr_batch(chunk)
        chunk_time = time.time() - chunk_start_time
        print(f"Processed batch {i//max_batch_size + 1} in {chunk_time:.2f}s ({len(chunk)} images, {chunk_time/len(chunk):.2f}s per image)")
        results.extend(chunk_results)

    total_time = time.time() - total_start_time
    print(f"Total processing time: {total_time:.2f}s ({len(tasks)} images, {total_time/len(tasks):.2f}s per image)")
    return results

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to") 
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)