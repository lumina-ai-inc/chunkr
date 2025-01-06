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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'

load_dotenv(dotenv_path=ENV_PATH)

batch_wait_time = float(os.getenv("BATCH_WAIT_TIME", 0.5))
max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 4))

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

class BoundingBoxOutput(BaseModel):
    left: float
    top: float
    width: float 
    height: float

class InstanceOutput(BaseModel):
    boxes: list[BoundingBoxOutput]
    scores: list[float]
    classes: list[int]
    image_size: tuple[int, int]
    
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

    
class OCRInput(BaseModel):
    bbox: BoundingBoxOutput
    text: str
    confidence: float | None = None


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

def create_grid_dict_from_ocr(ocr_results: List[OCRInput]) -> dict:
    if not ocr_results:
        return {
            "input_ids": [],
            "bbox_subword_list": [],
            "texts": [],
            "bbox_texts_list": []
        }
    texts = []
    boxes = []
    for result in ocr_results:
        texts.append(result.text)
        bbox = result.bbox
        boxes.append((bbox.left, bbox.top, bbox.width, bbox.height))
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
    def get_page_zones(segments_data: List[tuple]) -> dict:
        # Get y-coordinates for all segments
        y_centers = [(box, (box.y1 + box.y2) / 2) for box, score, cls in segments_data]
        y_coords = np.array([y[1] for y in y_centers]).reshape(-1, 1)
        
        # Use clustering to find major vertical zones
        n_clusters = min(3, len(y_coords))  # At most 3 zones (header, body, footer)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(y_coords)
        centers = kmeans.cluster_centers_

        # Sort zones by y-position
        sorted_zones = sorted(enumerate(centers), key=lambda x: x[1])
        zone_mapping = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(sorted_zones)}
        
        zones = {
            'header': [],
            'body': [],
            'footer': []
        }
        
        # Assign segments to zones
        for (box, y_center), label in zip(y_centers, labels):
            zone_idx = zone_mapping[label]
            if zone_idx == 0:
                zones['header'].append(box)
            elif zone_idx == len(centers) - 1:
                zones['footer'].append(box)
            else:
                zones['body'].append(box)
                
        return zones

    def process_body_content(body_segments: List[tuple]) -> List[tuple]:
        # Separate title area from main content
        y_coords = np.array([(box.y1 + box.y2) / 2 for box in body_segments]).reshape(-1, 1)
        if len(y_coords) > 1:
            title_clusters = KMeans(n_clusters=min(2, len(y_coords)), random_state=42)
            title_labels = title_clusters.fit_predict(y_coords)
            
            title_area = []
            main_content = []
            
            for segment, label in zip(body_segments, title_labels):
                if label == 0 and segment.y1 < np.median(y_coords):
                    title_area.append(segment)
                else:
                    main_content.append(segment)
                    
            # Process columns in main content
            columns = process_columns(main_content)
            return title_area + columns
        return body_segments

    def process_columns(segments: List[tuple]) -> List[tuple]:
        if not segments:
            return []

        # Get x-coordinates for column detection
        x_centers = [(seg, (seg.x1 + seg.x2) / 2) for seg in segments]
        x_coords = np.array([x[1] for x in x_centers]).reshape(-1, 1)
        
        # Determine optimal number of columns
        best_n_columns = 1
        best_score = -1
        
        for n_cols in range(1, min(4, len(x_coords) + 1)):
            kmeans = KMeans(n_clusters=n_cols, random_state=42)
            labels = kmeans.fit_predict(x_coords)
            if n_cols > 1:
                score = silhouette_score(x_coords, labels)
                if score > best_score:
                    best_score = best_score
                    best_n_columns = n_cols

        # Final column clustering
        kmeans = KMeans(n_clusters=best_n_columns, random_state=42)
        column_labels = kmeans.fit_predict(x_coords)
        column_centers = kmeans.cluster_centers_

        # Organize segments into columns
        columns = [[] for _ in range(best_n_columns)]
        for (segment, x_center), label in zip(x_centers, column_labels):
            # Check if segment spans multiple columns
            width = segment.x2 - segment.x1
            page_width = max(seg.x2 for seg in segments)
            is_wide_element = width > (page_width / best_n_columns) * 1.5
            
            if is_wide_element:
                # Place wide elements in leftmost affected column
                distances = [abs(x_center - center) for center in column_centers]
                nearest_col = distances.index(min(distances))
                columns[nearest_col].append(segment)
            else:
                columns[label].append(segment)

        # Sort within each column by y-coordinate
        for col in columns:
            col.sort(key=lambda seg: seg.y1)

        # Combine columns left to right
        return [seg for col in columns for seg in col]

    # Main processing
    for pred in predictions:
        segments = [(box, score, cls) for box, score, cls 
                   in zip(pred.instances.boxes, pred.instances.scores, pred.instances.classes)]
        
        # Get page zones
        zones = get_page_zones(segments)
        
        # Process each zone
        ordered_segments = []
        ordered_segments.extend(sorted(zones['header'], key=lambda x: x.y1))  # Header
        ordered_segments.extend(process_body_content(zones['body']))  # Body (including titles and columns)
        ordered_segments.extend(sorted(zones['footer'], key=lambda x: x.y1))  # Footer

        pred.instances.boxes = ordered_segments
        pred.instances.scores = [score for _, score, _ in segments]
        pred.instances.classes = [cls for _, _, cls in segments]

    return predictions

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
            raw_predictions = process_image_batch(
                predictor, 
                images, 
                dataset_name="doclaynet", 
                grid_dicts=grid_dicts_data
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("Detected 'out of memory' error - killing server.")
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
                    image_size=[instances.image_size[0], instances.image_size[1]]
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


class FinalPrediction(BaseModel):
    instances: InstanceOutput


@app.post("/batch_async", response_model=FinalPrediction)
async def create_od_task(
    file: UploadFile = File(...),
    ocr_data: str = Form(...)
):
    image_data = await file.read()
    if not ocr_data:
        ocr_words = []
    else:
        ocr_words = [OCRInput(**x) for x in json.loads(ocr_data)["data"]]
        
    grid_dict = create_grid_dict_from_ocr(ocr_words)
    future = asyncio.Future()
    task = ODTask(file_data=image_data, grid_dict=grid_dict, future=future)
    pending_tasks.append(task)
    batch_event.set()
    result = await future
    
    serializable_pred = result[0]
    converted_boxes = []
    for box in serializable_pred.instances.boxes:
        converted_boxes.append(BoundingBoxOutput(
            left=box.x1,
            top=box.y1,
            width=box.x2 - box.x1,
            height=box.y2 - box.y1
        ))
        
    final_pred = FinalPrediction(
        instances=InstanceOutput(
            boxes=converted_boxes,
            scores=serializable_pred.instances.scores,
            classes=serializable_pred.instances.classes,
            image_size=serializable_pred.instances.image_size
        )
    )
    
    return final_pred

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
