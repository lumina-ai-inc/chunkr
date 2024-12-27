import os
import time
import asyncio
from collections import deque
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Tuple
import json

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

# -------------
# Imports from your existing code
# -------------

from memory_inference import create_predictor, process_image_batch
from typing import Tuple

app = FastAPI()  # This will be changed below to a lifespan-based app for dynamic batching

# ------------------------------------------------
# Original global predictor (as in your code)
# ------------------------------------------------
predictor = None

CONFIG_FILE = "object_detection/configs/cascade/doclaynet_VGT_cascade_PTM.yaml"
WEIGHTS_PATH = "object_detection/weights/doclaynet_VGT_model.pth"

# ------------------------------------------------
# Dynamic batching config
# ------------------------------------------------
batch_wait_time = float(os.getenv("BATCH_WAIT_TIME", 0.5))
max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 4))

processing_lock = asyncio.Lock()
batch_event = asyncio.Event()
pending_tasks = deque()

# ------------------------------------------------
# Pydantic models for dynamic batching
# ------------------------------------------------
class Grid(BaseModel):
    input_ids: List[int]
    bbox_subword_list: List[List[float]]
    texts: List[str]
    bbox_texts_list: List[List[float]]

class GridDicts(BaseModel):
    grid_dicts: List[Grid]

class Instance(BaseModel):
    boxes: List[List[float]]
    scores: List[float]
    classes: List[int]
    image_size: Tuple[int, int]

class SerializablePrediction(BaseModel):
    instances: Instance

# Task object to store everything needed for the batch
class ODTask(BaseModel):
    file_data: bytes       # The raw bytes of the image
    grid_dict: dict        # The grid dict object
    future: asyncio.Future # Will hold the result once processed

    class Config:
        arbitrary_types_allowed = True

# A convenience model for returning the result or any custom shape
class ODResponse(BaseModel):
    predictions: List[SerializablePrediction]

# ------------------------------------------------
# Helper functions (copied from your existing code)
# ------------------------------------------------

def get_reading_order(predictions: List[SerializablePrediction]) -> List[SerializablePrediction]:
    type_mapping = {
        0: "Caption", 
        1: "Footnote", 
        2: "Formula", 
        3: "List-item", 
        4: "Page-footer", 
        5: "Page-header", 
        6: "Picture", 
        7: "Section-header", 
        8: "Table", 
        9: "Text", 
        10: "Title"
    }

    def get_segment_type(class_id: int) -> str:
        return type_mapping.get(class_id, "Text")

    def sort_segments_by_position(segments_data: List[tuple]) -> List[tuple]:
        vertical_threshold = 50
        def sort_key(segment):
            box, _, _ = segment
            y = box[1]
            x = box[0]
            return (y // vertical_threshold, x)
        return sorted(segments_data, key=sort_key)

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
            if segment_type == "Page-header":
                header_segments.append(segment_data)
            elif segment_type == "Page-footer":
                footer_segments.append(segment_data)
            else:
                main_segments.append(segment_data)
        
        header_segments = sort_segments_by_position(header_segments)
        main_segments = sort_segments_by_position(main_segments)
        footer_segments = sort_segments_by_position(footer_segments)
        
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

def merge_colliding_predictions(boxes: List[List[float]], scores: List[float], classes: List[int]) -> tuple[List[List[float]], List[float], List[int]]:
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
                    max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) *
                    max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                )
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
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
                
                x1 = min(box[0] for box in merge_boxes)
                y1 = min(box[1] for box in merge_boxes)
                x2 = max(box[2] for box in merge_boxes)
                y2 = max(box[3] for box in merge_boxes)
                
                # If any of them is table => class 8
                if 8 in merge_classes:
                    final_class = 8
                else:
                    max_score_idx = merge_scores.index(max(merge_scores))
                    final_class = merge_classes[max_score_idx]
                
                new_boxes.append([x1, y1, x2, y2])
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
            text_box = [x, y, x + w, y + h]
            
            best_score = 0
            best_class = 9  # default to "Text"
            
            for box, score, cls in zip(boxes, scores, classes):
                intersection = (
                    max(0, min(text_box[2], box[2]) - max(text_box[0], box[0])) *
                    max(0, min(text_box[3], box[3]) - max(text_box[1], box[1]))
                )
                text_area = (text_box[2] - text_box[0]) * (text_box[3] - text_box[1])
                pred_area = (box[2] - box[0]) * (box[3] - box[1])
                min_area = min(text_area, pred_area)
                
                if min_area > 0 and intersection / min_area > 0.1 and score > best_score:
                    best_score = score
                    best_class = cls
            
            text_boxes.append(text_box)
            text_scores.append(best_score)
            text_classes.append(best_class)
        
        # Include predictions without associated text
        for box, score, cls in zip(boxes, scores, classes):
            has_overlap = False
            for text_box in text_boxes:
                intersection = (
                    max(0, min(text_box[2], box[2]) - max(text_box[0], box[0])) *
                    max(0, min(text_box[3], box[3]) - max(text_box[1], box[1]))
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

# ------------------------------------------------
# Dynamic Batching Logic
# ------------------------------------------------
async def process_od_batch(tasks: List[ODTask]) -> List[List[SerializablePrediction]]:
    """
    Processes a list of ODTask in a single batch. This merges all images & grid dicts
    and keeps track of each in order to properly split the results back out.
    """
    # Flatten out all images from tasks:
    images = []
    grid_dicts_data = []
    for t in tasks:
        image = cv2.imdecode(np.frombuffer(t.file_data, np.uint8), cv2.IMREAD_COLOR)
        images.append(image)
        grid_dicts_data.append(t.grid_dict)  # Each task presumably has one

    # The process_image_batch from your existing code
    # Here, we pass a list of grid_dicts. If each task has exactly 1, you pass them individually.
    # If each task can have multiple, you'll need to flatten them differently.
    raw_predictions, _ = process_image_batch(
        predictor, 
        images, 
        dataset_name="doclaynet", 
        grid_dicts=grid_dicts_data
    )

    # Convert raw predictions to serializable form
    predictions_list = []
    for pred in raw_predictions:
        serializable_pred = SerializablePrediction(
            instances=Instance(
                boxes=pred["instances"].pred_boxes.tensor.tolist(),
                scores=pred["instances"].scores.tolist(),
                classes=pred["instances"].pred_classes.tolist(),
                image_size=[
                    pred["instances"].image_size[0],
                    pred["instances"].image_size[1]
                ]
            )
        )
        predictions_list.append(serializable_pred)

    # For each input in tasks, find the best segments -> reading order
    # Because tasks and predictions_list are in the same order, pair them up:
    results_for_tasks = []
    for i, t in enumerate(tasks):
        # Wrap the single grid_dict as a GridDicts object
        grid_dicts_obj = GridDicts(grid_dicts=[t.grid_dict])
        final_predictions = find_best_segments([predictions_list[i]], grid_dicts_obj)
        ordered_predictions = get_reading_order(final_predictions)
        results_for_tasks.append(ordered_predictions)  # Each task yields a list of predictions

    return results_for_tasks

async def batch_processor():
    """
    The background task that waits for new items in the queue
    and processes them in a batch after a short wait.
    """
    while True:
        await batch_event.wait()
        batch_event.clear()
        
        # Wait for any straggler tasks to come in
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
                # If you want to break them up by max_batch_size
                for i in range(0, len(current_batch), max_batch_size):
                    chunk = current_batch[i:i+max_batch_size]
                    chunk_results = await process_od_batch(chunk)
                    results.extend(chunk_results)

            # Assign each result to the future in its corresponding task
            for task, result in zip(current_batch, results):
                task.future.set_result(result)
        except Exception as e:
            for task in current_batch:
                task.future.set_exception(e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown code. We also load the model here 
    and start the background batch_processor.
    """
    global predictor
    
    # Create predictor if not already created
    if predictor is None:
        predictor = create_predictor(CONFIG_FILE, WEIGHTS_PATH)
        print("Model loaded successfully!")
    
    batch_processor_task = asyncio.create_task(batch_processor())    
    yield
    # On shutdown
    batch_processor_task.cancel()

# Create a new FastAPI app with a lifespan that starts the background processor
app = FastAPI(lifespan=lifespan)

# ------------------------------------------------
# Dynamic Batching Endpoint
# ------------------------------------------------
@app.post("/batch_async", response_model=SerializablePrediction)
async def create_od_task(
    file: UploadFile = File(...),
    grid_dict: str = Form(...)
):
    """
    A single-image endpoint that will queue up tasks for dynamic batching.
    """
    image_data = await file.read()
    grid_dict_data = json.loads(grid_dict)

    # Create the task and attach a Future
    future = asyncio.Future()
    task = ODTask(file_data=image_data, grid_dict=grid_dict_data, future=future)

    pending_tasks.append(task)
    batch_event.set()

    # Wait until the batch is processed
    result = await future  # This will be a list of predictions
    # Return just the first prediction (since this is a single-image endpoint)
    return result[0] # Changed from result[0]

# ------------------------------------------------
# (Optional) Keep your old endpoints or adapt them
# ------------------------------------------------
@app.post("/batch/")
async def process_image_batch_endpoint(
    files: List[UploadFile] = File(...),
    grid_dicts: str = Form(...)
):
    """
    Your existing endpoint. You may keep it or remove it. 
    In a pure dynamic batching scenario, you'd rely on /batch_async or a loop of single calls.
    """
    from fastapi.responses import JSONResponse

    grid_dicts_data = json.loads(grid_dicts)
    grid_dicts_obj = GridDicts(grid_dicts=grid_dicts_data)

    images = []
    for file in files:
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        images.append(image)
    
    raw_predictions, _ = process_image_batch(
        predictor, images, dataset_name="doclaynet", grid_dicts=grid_dicts_data
    )
    
    predictions = []
    for pred in raw_predictions:
        serializable_pred = SerializablePrediction(
            instances=Instance(
                boxes=pred["instances"].pred_boxes.tensor.tolist(),
                scores=pred["instances"].scores.tolist(),
                classes=pred["instances"].pred_classes.tolist(),
                image_size=[
                    pred["instances"].image_size[0],
                    pred["instances"].image_size[1]
                ]
            )
        )
        predictions.append(serializable_pred)
    
    final_predictions = find_best_segments(predictions, grid_dicts_obj)
    ordered_predictions = get_reading_order(final_predictions)
    
    return JSONResponse(content=[p.dict() for p in ordered_predictions])

@app.post("/single/")
async def process_single_image_endpoint(
    file: UploadFile = File(...),
    grid_dict: str = Form(...)
):
    """
    Your existing single-image endpoint. 
    """
    from fastapi.responses import JSONResponse

    grid_dict_data = json.loads(grid_dict)
    grid_dicts_obj = GridDicts(grid_dicts=[grid_dict_data])

    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    raw_predictions, _ = process_image_batch(
        predictor, [image], dataset_name="doclaynet", grid_dicts=[grid_dict_data]
    )
    
    pred = raw_predictions[0]
    serializable_pred = SerializablePrediction(
        instances=Instance(
            boxes=pred["instances"].pred_boxes.tensor.tolist(),
            scores=pred["instances"].scores.tolist(),
            classes=pred["instances"].pred_classes.tolist(),
            image_size=[
                pred["instances"].image_size[0],
                pred["instances"].image_size[1]
            ]
        )
    )

    final_predictions = find_best_segments([serializable_pred], grid_dicts_obj)
    ordered_predictions = get_reading_order(final_predictions)
    
    return JSONResponse(content=ordered_predictions[0].dict())

@app.get("/")
async def root():
    return {"message": "Hello World"}

# In main mode, run uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)