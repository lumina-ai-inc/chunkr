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
import psutil
import collections
import uuid
from pydantic import Field
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
# from configuration import MODELS_PATH
import gc


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'

load_dotenv(dotenv_path=ENV_PATH)

batch_wait_time = float(os.getenv("BATCH_WAIT_TIME", 0.1))
max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 4))
overlap_threshold = float(os.getenv("OVERLAP_THRESHOLD", 0.1))
score_threshold = float(os.getenv("SCORE_THRESHOLD", 0.2))
print(f"Max batch size: {max_batch_size}")
print(f"Overlap threshold: {overlap_threshold}")

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
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

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

def apply_reading_order(instances):
    bxs = instances.get("boxes", [])
    scs = instances.get("scores", [])
    cls = instances.get("classes", [])
    if not bxs:
        return instances

    def is_wide_element(box, page_width, threshold=0.7):
        return box["width"] / page_width > threshold

    def get_column_assignment(box, col_boundaries):
        center_x = box["left"] + box["width"] / 2
        for i, (left, right) in enumerate(col_boundaries):
            if left <= center_x <= right:
                return i
        return 0

    # Calculate page width
    page_width = max(box["left"] + box["width"] for box in bxs) if bxs else 1000

    # Determine column boundaries using clustering
    centers_x = np.array([[box["left"] + box["width"] / 2] for box in bxs])
    
    # Use the elbow method to find the optimal number of clusters
    distortions = []
    K = range(1, min(10, len(centers_x) + 1))  # Ensure K is within valid range
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(centers_x)
        distortions.append(kmeans.inertia_)

    # Find the elbow point
    optimal_k = 1
    for i in range(1, len(distortions) - 1):
        if distortions[i] - distortions[i + 1] < distortions[i - 1] - distortions[i]:
            optimal_k = i + 1
            break

    # Perform clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k)
    kmeans.fit(centers_x)
    col_boundaries = []
    for i in range(optimal_k):
        cluster_points = centers_x[kmeans.labels_ == i]
        if cluster_points.size > 0:
            col_boundaries.append((cluster_points.min(), cluster_points.max()))

    # Ensure col_boundaries is not empty and sort them
    if not col_boundaries:
        col_boundaries = [(0, page_width)]
    col_boundaries.sort(key=lambda x: x[0])  # Sort by the left boundary

    # Process segments
    segments = []
    for i, (box, score, class_id) in enumerate(zip(bxs, scs, cls)):
        if score > 0.2:
            is_wide = is_wide_element(box, page_width)
            segments.append({
                'idx': i,
                'box': box,
                'score': score,
                'class': class_id,
                'is_wide': is_wide,
                'center_x': box["left"] + box["width"] / 2,
                'center_y': box["top"] + box["height"] / 2
            })

    # Separate headers and footers
    headers = [s for s in segments if s['class'] == 5]
    footers = [s for s in segments if s['class'] in [1, 4]]
    body = [s for s in segments if s['class'] not in [1, 4, 5]]

    def process_body_segments(segments):
        segments.sort(key=lambda s: s['box']['top'])
        columns = [[] for _ in range(optimal_k)]
        current_y = float('-inf')
        temp_segments = []

        for segment in segments:
            if abs(segment['box']['top'] - current_y) > 20:
                if temp_segments:
                    if any(s['is_wide'] for s in temp_segments):
                        for s in temp_segments:
                            s['is_wide'] = True
                    for s in temp_segments:
                        if s['is_wide']:
                            for col in columns:
                                yield from col
                            columns = [[] for _ in range(optimal_k)]
                            yield s
                        else:
                            col = get_column_assignment(s['box'], col_boundaries)
                            columns[col].append(s)
                temp_segments = [segment]
                current_y = segment['box']['top']
            else:
                temp_segments.append(segment)

        if temp_segments:
            if any(s['is_wide'] for s in temp_segments):
                for col in columns:
                    yield from col
                yield from temp_segments
            else:
                for s in temp_segments:
                    col = get_column_assignment(s['box'], col_boundaries)
                    columns[col].append(s)

        for col in columns:
            yield from col

    ordered_segments = []
    ordered_segments.extend(sorted(headers, key=lambda s: s['box']['top']))
    ordered_segments.extend(process_body_segments(body))
    ordered_segments.extend(sorted(footers, key=lambda s: s['box']['top']))

    if ordered_segments:
        final_indices = [s['idx'] for s in ordered_segments]
        instances["boxes"] = [bxs[i] for i in final_indices]
        instances["scores"] = [scs[i] for i in final_indices]
        instances["classes"] = [cls[i] for i in final_indices]

    return instances

def get_reading_order(predictions: List[SerializablePrediction]) -> List[SerializablePrediction]:
    def convert_to_dict_format(boxes, scores, classes, image_size):
        dict_boxes = []
        for box in boxes:
            dict_boxes.append({
                "left": box.x1,
                "top": box.y1,
                "width": box.x2 - box.x1,
                "height": box.y2 - box.y1
            })
        return {
            "boxes": dict_boxes,
            "scores": scores,
            "classes": classes,
            "image_size": image_size
        }

    def convert_from_dict_format(instances_dict):
        boxes = []
        for box in instances_dict["boxes"]:
            boxes.append(BoundingBox(
                x1=box["left"],
                y1=box["top"],
                x2=box["left"] + box["width"],
                y2=box["top"] + box["height"]
            ))
        return boxes, instances_dict["scores"], instances_dict["classes"]

    updated_predictions = []
    for pred in predictions:
        instances_dict = convert_to_dict_format(
            pred.instances.boxes,
            pred.instances.scores,
            pred.instances.classes,
            pred.instances.image_size
        )
        
        ordered_dict = apply_reading_order(instances_dict)
        
        boxes, scores, classes = convert_from_dict_format(ordered_dict)
        
        ordered_pred = SerializablePrediction(
            instances=Instance(
                boxes=boxes,
                scores=scores,
                classes=classes,
                image_size=pred.instances.image_size
            )
        )
        updated_predictions.append(ordered_pred)
    
    return updated_predictions


def merge_colliding_predictions(boxes: List[BoundingBox], scores: List[float], classes: List[int]) -> tuple[List[BoundingBox], List[float], List[int]]:
    valid_indices = [i for i, score in enumerate(scores) if score >= score_threshold]
    if not valid_indices:
        return boxes, scores, classes
    
    filtered_boxes = [boxes[i] for i in valid_indices]
    filtered_scores = [scores[i] for i in valid_indices]
    filtered_classes = [classes[i] for i in valid_indices]
    
    merged_boxes, merged_scores, merged_classes = [], [], []
    
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
            
            if min_area > 0 and (intersection / min_area) > overlap_threshold:
                to_merge_indices.append(i)
        
        if to_merge_indices:
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
            
            class_counts = collections.Counter(merge_classes)
            final_class = class_counts.most_common(1)[0][0]
            
            merged_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
            merged_scores.append(max(merge_scores))
            merged_classes.append(final_class)
        else:
            merged_boxes.append(box1)
            merged_scores.append(score1)
            merged_classes.append(class1)
    
    # Merge small boxes into larger ones if overlap > 90%
    final_boxes, final_scores, final_classes = [], [], []
    for i, box1 in enumerate(merged_boxes):
        merged = False
        for j, box2 in enumerate(merged_boxes):
            if i != j:
                intersection = (
                    max(0, min(box1.x2, box2.x2) - max(box1.x1, box2.x1)) *
                    max(0, min(box1.y2, box2.y2) - max(box1.y1, box2.y1))
                )
                area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
                if area1 > 0 and (intersection / area1) > 0.9:
                    merged = True
                    break
        if not merged:
            final_boxes.append(box1)
            final_scores.append(merged_scores[i])
            final_classes.append(merged_classes[i])
    
    return final_boxes, final_scores, final_classes


def find_best_segments(predictions: List[SerializablePrediction], grid_dicts: GridDicts) -> List[SerializablePrediction]:
    grid_list = grid_dicts.grid_dicts if isinstance(grid_dicts, GridDicts) else grid_dicts
    
    updated_predictions = []
    for pred, grid in zip(predictions, grid_list):
        boxes, scores, classes = merge_colliding_predictions(
            pred.instances.boxes,
            pred.instances.scores,
            pred.instances.classes
        )
        
        updated_pred = SerializablePrediction(
            instances=Instance(
                boxes=boxes,
                scores=scores,
                classes=classes,
                image_size=pred.instances.image_size
            )
        )
        updated_predictions.append(updated_pred)
    
    return updated_predictions


async def decode_image(file_data):
    return cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)

async def preprocess_batch(tasks: List[ODTask]) -> Tuple[List[np.ndarray], List[dict]]:
    images = await asyncio.gather(*[decode_image(t.file_data) for t in tasks])
    grid_dicts = [t.grid_dict for t in tasks]
    return images, grid_dicts

async def run_inference(images: List[np.ndarray], grid_dicts: List[dict]) -> List[dict]:
    
    import sys
    try:
        predictions = process_image_batch(
            predictor,
            images,
            dataset_name="doclaynet",
            grid_dicts=grid_dicts
        )
        return predictions
    except Exception as e:
        if "out of memory" in str(e).lower():
            print("Detected 'out of memory' error - killing server.")
            kill_server()
            sys.exit(1)
        raise e
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def postprocess_predictions(raw_predictions: List[dict], grid_dicts: List[dict]) -> List[List[SerializablePrediction]]:
    predictions_list = []
    for pred in raw_predictions:
        instances = pred["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.tolist()
        
        serializable_pred = SerializablePrediction(
            instances=Instance(
                boxes=[BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3]) for b in boxes],
                scores=instances.scores.tolist(),
                classes=instances.pred_classes.tolist(),
                image_size=list(instances.image_size)
            )
        )
        predictions_list.append(serializable_pred)

    results = []
    for pred, grid_dict in zip(predictions_list, grid_dicts):
        grid_dicts_obj = GridDicts(grid_dicts=[grid_dict])
        final_predictions = find_best_segments([pred], grid_dicts_obj)
        ordered_predictions = get_reading_order(final_predictions)
        results.append(ordered_predictions)
    
    return results

async def process_od_batch(tasks: List[ODTask]) -> List[List[SerializablePrediction]]:
    print(f"Processing batch of {len(tasks)} tasks | {time.time()}")
    
    # Phase 1: Preprocessing entire batch
    images, grid_dicts = await preprocess_batch(tasks)
    
    # Phase 2: GPU Inference in sub-batches
    raw_predictions = []
    for i in range(0, len(tasks), max_batch_size):
        sub_batch_images = images[i:i + max_batch_size]
        sub_batch_grids = grid_dicts[i:i + max_batch_size]
        print(f"Processing task IDs: {[task.task_id for task in tasks[i:i + max_batch_size]]}")
        
        sub_batch_predictions = await run_inference(sub_batch_images, sub_batch_grids)
        raw_predictions.extend(sub_batch_predictions)
        
        # Cleanup after each inference
        del sub_batch_images
        gc.collect()
    
    # Phase 3: Postprocessing entire batch
    results = postprocess_predictions(raw_predictions, grid_dicts)
    
    # Final cleanup
    del images, raw_predictions
    gc.collect()
    
    return results

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
    print("Starting server... Max batch size: ", max_batch_size)
    uvicorn.run(app, host="0.0.0.0", port=8000)
