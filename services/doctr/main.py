import asyncio
from collections import deque
from contextlib import asynccontextmanager
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, db_resnet50, master, vitstr_small
import dotenv
from fastapi import FastAPI, File, UploadFile
import numpy as np
import os
from pydantic import BaseModel
from pydantic.generics import GenericModel
import time
import torch
from typing import Dict, List, TypeVar, Generic, Optional
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Disable compilation altogether for now
torch.compile = lambda x, *args, **kwargs: x
dotenv.load_dotenv(override=True)

batch_wait_time = float(os.getenv('OCR_BATCH_WAIT_TIME', 0.25))
max_batch_size = int(os.getenv('OCR_MAX_BATCH_SIZE', 50))
detection_model = db_resnet50(pretrained=True).eval()
recognition_model = vitstr_small(pretrained=True).eval()

predictor = ocr_predictor(detection_model, recognition_model, pretrained=True, 
                         export_as_straight_boxes=True)
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("Using GPU")
    predictor = predictor.cuda().half()
else:
    print("Using CPU")

os.environ['USE_TORCH'] = 'YES'
os.environ['USE_TF'] = 'NO'
pending_tasks = deque()
processing_lock = asyncio.Lock()
batch_event = asyncio.Event()
server_processing_lock = asyncio.Lock()

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
    image_bytes_list = [task.image_data for task in tasks]
    
    # Process image loading concurrently
    doc = await asyncio.get_event_loop().run_in_executor(
        None, DocumentFile.from_images, image_bytes_list
    )
    
    start_time = time.time()
    async with server_processing_lock:
        with torch.inference_mode(), torch.amp.autocast('cuda'):
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)


@app.post("/batch")
async def batch_ocr(files: List[UploadFile] = File(...)):
    total_start_time = time.time()
    tasks = []
    for file in files:
        image_data = await file.read()
        task = OCRTask(image_data=image_data)
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