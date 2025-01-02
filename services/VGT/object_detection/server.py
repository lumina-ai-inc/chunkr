import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from typing import List, Tuple
import time
import json
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from memory_inference import create_predictor, process_image_batch
from fastapi.responses import JSONResponse
import torch
import signal
import psutil

from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'

load_dotenv(dotenv_path=ENV_PATH)

batch_wait_time = float(os.getenv("BATCH_WAIT_TIME", 0.5))
max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 8))

print(f"Max batch size: {max_batch_size}")

app = FastAPI()  

predictor = None

CONFIG_FILE = "object_detection/configs/cascade/doclaynet_VGT_cascade_PTM.yaml"
WEIGHTS_PATH = "object_detection/weights/doclaynet_VGT_model.pth"

processing_lock = asyncio.Lock()
batch_event = asyncio.Event()
pending_tasks = deque()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def kill_server():
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    current_process.kill()

class Grid(BaseModel):
    input_ids: List[int]
    bbox_subword_list: List[List[float]]
    texts: List[str]
    bbox_texts_list: List[List[float]]

class GridDicts(BaseModel):
    grid_dicts: List[Grid]

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Instance(BaseModel):
    boxes: List[BoundingBox]
    scores: List[float]
    classes: List[int]
    image_size: Tuple[int, int]

class SerializablePrediction(BaseModel):
    instances: Instance

class ODTask(BaseModel):
    file_data: bytes
    grid_dict: dict
    future: asyncio.Future

    class Config:
        arbitrary_types_allowed = True

class ODResponse(BaseModel):
    predictions: List[SerializablePrediction]

class OcrWord(BaseModel):
    left: float
    top: float
    width: float
    height: float
    text: str
    confidence: float = 1.0

def tokenize_texts(text_body: List[str]) -> List[List[int]]:
    tokenized = tokenizer.batch_encode_plus(
        text_body,
        return_token_type_ids=False,
        return_attention_mask=False,
        add_special_tokens=False
    )
    return tokenized["input_ids"]

def readjust_bbox_coords(bounding_boxes: List[Tuple[float,float,float,float]], tokens: List[List[int]]) -> List[Tuple[float,float,float,float]]:
    adjusted = []
    for box, sub_tokens in zip(bounding_boxes, tokens):
        if len(sub_tokens) > 1:
            new_width = box[2] / len(sub_tokens)
            for i in range(len(sub_tokens)):
                adjusted.append((box[0] + i * new_width, box[1], new_width, box[3]))
        else:
            adjusted.append(box)
    return adjusted

def create_grid_dict_from_ocr(ocr_words: List[OcrWord]) -> dict:
    if not ocr_words:
        return {
            "input_ids": [],
            "bbox_subword_list": [],
            "texts": [],
            "bbox_texts_list": []
        }
    texts = []
    boxes = []
    for w in ocr_words:
        texts.append(w.text)
        boxes.append((w.left, w.top, w.width, w.height))
    sub_tokens = tokenize_texts(texts)
    input_ids = []
    for st in sub_tokens:
        input_ids.extend(st)
    subword_bboxes = readjust_bbox_coords(boxes, sub_tokens)
    return {
        "input_ids": input_ids,
        "bbox_subword_list": subword_bboxes,
        "texts": texts,
        "bbox_texts_list": boxes
    }

def get_reading_order(predictions: List[SerializablePrediction]) -> List[SerializablePrediction]:
    type_mapping = {
        0: "Caption", 
        1: "Footnote", 
        2: "Formula", 
        3: "ListItem", 
        4: "PageFooter", 
        5: "PageHeader", 
        6: "Picture", 
        7: "SectionHeader", 
        8: "Table", 
        9: "Text", 
        10: "Title"
    }

    def get_segment_type(class_id: int) -> str:
        return type_mapping.get(class_id, "Text")

    def columns_layout_sort(segments_data: List[tuple]) -> List[tuple]:
        if not segments_data:
            return []

        horizontal_threshold = 100

        segments_data = sorted(segments_data, key=lambda seg: seg[0].x1)

        columns = []
        for seg in segments_data:
            box, score, cls = seg
            placed = False
            for col_index, col_items in enumerate(columns):
                col_x1s = [item[0].x1 for item in col_items]
                col_x2s = [item[0].x2 for item in col_items]
                col_min_x = min(col_x1s)
                col_max_x = max(col_x2s)

                if abs(box.x1 - col_min_x) < horizontal_threshold or abs(box.x1 - col_max_x) < horizontal_threshold:
                    columns[col_index].append(seg)
                    placed = True
                    break

            if not placed:
                columns.append([seg])

        column_positions = []
        for col_index, col_items in enumerate(columns):
            col_items.sort(key=lambda item: (item[0].y1, item[0].x1))
            avg_x_positions = [it[0].x1 for it in col_items]
            column_positions.append((col_index, sum(avg_x_positions) / len(avg_x_positions)))

        column_positions.sort(key=lambda cp: cp[1])
        sorted_columns = []
        for col_index, _ in column_positions:
            sorted_columns.extend(columns[col_index])

        return sorted_columns

    updated_predictions = []
    
    for pred in predictions:
        header_segments = []
        main_segments = []
        footer_segments = []

        for box, score, cls in zip(pred.instances.boxes, 
                                   pred.instances.scores, 
                                   pred.instances.classes):
            segment_type = get_segment_type(cls)
            segment_data = (box, score, cls)
            if segment_type == "PageHeader":
                header_segments.append(segment_data)
            elif segment_type == "PageFooter":
                footer_segments.append(segment_data)
            else:
                main_segments.append(segment_data)
     
        header_segments.sort(key=lambda seg: (seg[0].y1, seg[0].x1))
        footer_segments.sort(key=lambda seg: (seg[0].y1, seg[0].x1))

        main_segments = columns_layout_sort(main_segments)

        ordered_segments = header_segments + main_segments + footer_segments

        if ordered_segments:
            boxes, scores, classes = zip(*ordered_segments)
        else:
            boxes, scores, classes = [], [], []
        
        ordered_pred = SerializablePrediction(
            instances=Instance(
                boxes=list(boxes),
                scores=list(scores),
                classes=list(classes),
                image_size=pred.instances.image_size
            )
        )
        updated_predictions.append(ordered_pred)
    
    return updated_predictions

def merge_colliding_predictions(boxes: List[BoundingBox], scores: List[float], classes: List[int]) -> tuple[List[BoundingBox], List[float], List[int]]:
    valid_indices = [i for i, score in enumerate(scores) if score >= 0.2]
    if not valid_indices:
        return boxes, scores, classes
    
    filtered_boxes = [boxes[i] for i in valid_indices]
    filtered_scores = [scores[i] for i in valid_indices]
    filtered_classes = [classes[i] for i in valid_indices]
    
    while True:
        merged = False
        new_boxes, new_scores, new_classes = [], [], []
        
        while filtered_boxes:
            box1 = filtered_boxes.pop(0)
            score1 = filtered_scores.pop(0)
            class1 = filtered_classes.pop(0)
            
            to_merge_indices = []
            for i, box2 in enumerate(filtered_boxes):
                intersection = (
                    max(0, min(box1.x2, box2.x2) - max(box1.x1, box2.x1)) *
                    max(0, min(box1.y2, box2.y2) - max(box1.y1, box2.y1))
                )
                area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
                area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
                min_area = min(area1, area2)
                
                if min_area > 0 and (intersection / min_area) > 0.1:
                    to_merge_indices.append(i)
            
            if to_merge_indices:
                merged = True
                merge_boxes = [box1] + [filtered_boxes[i] for i in to_merge_indices]
                merge_scores = [score1] + [filtered_scores[i] for i in to_merge_indices]
                merge_classes = [class1] + [filtered_classes[i] for i in to_merge_indices]
                
                for i in reversed(to_merge_indices):
                    filtered_boxes.pop(i)
                    filtered_scores.pop(i)
                    filtered_classes.pop(i)
                
                x1 = min(box.x1 for box in merge_boxes)
                y1 = min(box.y1 for box in merge_boxes)
                x2 = max(box.x2 for box in merge_boxes)
                y2 = max(box.y2 for box in merge_boxes)
                
                if 8 in merge_classes:
                    final_class = 8
                else:
                    max_score_idx = merge_scores.index(max(merge_scores))
                    final_class = merge_classes[max_score_idx]
                
                new_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
                new_scores.append(max(merge_scores))
                new_classes.append(final_class)
            else:
                new_boxes.append(box1)
                new_scores.append(score1)
                new_classes.append(class1)
        
        if not merged:
            return new_boxes, new_scores, new_classes
        
        filtered_boxes = new_boxes
        filtered_scores = new_scores
        filtered_classes = new_classes

def find_best_segments(predictions: List[SerializablePrediction], grid_dicts: GridDicts) -> List[SerializablePrediction]:
    grid_list = grid_dicts.grid_dicts if isinstance(grid_dicts, GridDicts) else grid_dicts
    
    updated_predictions = []
    for pred, grid in zip(predictions, grid_list):
        boxes, scores, classes = merge_colliding_predictions(
            pred.instances.boxes,
            pred.instances.scores,
            pred.instances.classes
        )
        text_boxes = []
        text_scores = []
        text_classes = []
        
        for text, bbox in zip(grid.texts, grid.bbox_texts_list):
            x, y, w, h = bbox
            text_box = BoundingBox(x1=x, y1=y, x2=x+w, y2=y+h)
            
            best_score = 0
            best_class = 9
            
            for box, score, cls in zip(boxes, scores, classes):
                intersection = (
                    max(0, min(text_box.x2, box.x2) - max(text_box.x1, box.x1)) *
                    max(0, min(text_box.y2, box.y2) - max(text_box.y1, box.y1))
                )
                text_area = (text_box.x2 - text_box.x1) * (text_box.y2 - text_box.y1)
                pred_area = (box.x2 - box.x1) * (box.y2 - box.y1)
                min_area = min(text_area, pred_area)
                
                if min_area > 0 and intersection / min_area > 0.1 and score > best_score:
                    best_score = score
                    best_class = cls
            
            text_boxes.append(text_box)
            text_scores.append(best_score)
            text_classes.append(best_class)
        
        for box, score, cls in zip(boxes, scores, classes):
            has_overlap = False
            for text_box in text_boxes:
                intersection = (
                    max(0, min(text_box.x2, box.x2) - max(text_box.x1, box.x1)) *
                    max(0, min(text_box.y2, box.y2) - max(text_box.y1, box.y1))
                )
                if intersection > 0:
                    has_overlap = True
                    break
            if not has_overlap:
                text_boxes.append(box)
                text_scores.append(score)
                text_classes.append(cls)
        
        updated_pred = SerializablePrediction(
            instances=Instance(
                boxes=text_boxes,
                scores=text_scores,
                classes=text_classes,
                image_size=pred.instances.image_size
            )
        )
        updated_predictions.append(updated_pred)
    
    return updated_predictions


async def process_od_batch(tasks: List[ODTask]) -> List[List[SerializablePrediction]]:
    print(f"Processing batch of {len(tasks)} tasks | {time.time()}")
    try:
        images = []
        grid_dicts_data = []
        for t in tasks:
            image = cv2.imdecode(np.frombuffer(t.file_data, np.uint8), cv2.IMREAD_COLOR)
            images.append(image)
            grid_dicts_data.append(t.grid_dict)  

        try:
            raw_predictions= process_image_batch(
                predictor, 
                images, 
                dataset_name="doclaynet", 
                grid_dicts=grid_dicts_data
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory error detected - killing server")
                kill_server()
                return
            raise e

        predictions_list = []
        for pred in raw_predictions:
            instances = pred["instances"].to("cpu")
            serializable_pred = SerializablePrediction(
                instances=Instance(
                    boxes=[BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]) 
                           for box in instances.pred_boxes.tensor.tolist()],
                    scores=instances.scores.tolist(),
                    classes=instances.pred_classes.tolist(),
                    image_size=[
                        instances.image_size[0],
                        instances.image_size[1]
                    ]
                )
            )
            predictions_list.append(serializable_pred)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results_for_tasks = []
        for i, t in enumerate(tasks):
            grid_dicts_obj = GridDicts(grid_dicts=[t.grid_dict])
            final_predictions = find_best_segments([predictions_list[i]], grid_dicts_obj)
            ordered_predictions = get_reading_order(final_predictions)
            results_for_tasks.append(ordered_predictions)  

        return results_for_tasks

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

async def batch_processor():
    while True:
        await batch_event.wait()
        batch_event.clear()
        
        await asyncio.sleep(batch_wait_time)

        async with processing_lock:
            if not pending_tasks:
                continue
            current_batch = list(pending_tasks)
            pending_tasks.clear()
        
        try:
            results = []
            if max_batch_size is None or max_batch_size <= 0:
                results.extend(await process_od_batch(current_batch))
            else:
                for i in range(0, len(current_batch), max_batch_size):
                    chunk = current_batch[i:i+max_batch_size]
                    chunk_results = await process_od_batch(chunk)
                    results.extend(chunk_results)

            for task, result in zip(current_batch, results):
                task.future.set_result(result)
        except Exception as e:
            for task in current_batch:
                task.future.set_exception(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    
    if predictor is None:
        predictor = create_predictor(CONFIG_FILE, WEIGHTS_PATH)
        print("Model loaded successfully!")
    
    batch_processor_task = asyncio.create_task(batch_processor())    
    yield
    batch_processor_task.cancel()

app = FastAPI(lifespan=lifespan)


@app.post("/batch_async", response_model=SerializablePrediction)
async def create_od_task(
    file: UploadFile = File(...),
    ocr_data: str = Form(...)
):
    image_data = await file.read()
    if not ocr_data:
        ocr_words = []
    else:
        ocr_words = [OcrWord(**x) for x in json.loads(ocr_data)]
    grid_dict = create_grid_dict_from_ocr(ocr_words)
    future = asyncio.Future()
    task = ODTask(file_data=image_data, grid_dict=grid_dict, future=future)
    pending_tasks.append(task)
    batch_event.set()
    result = await future  

    return result[0] 

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
