from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from memory_inference import create_predictor, process_image_batch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import json
app = FastAPI()

# Global variables
predictor = None

# Configuration paths
CONFIG_FILE = "object_detection/configs/cascade/doclaynet_VGT_cascade_PTM.yaml"
WEIGHTS_PATH = "object_detection/weights/doclaynet_VGT_model.pth"

@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = create_predictor(CONFIG_FILE, WEIGHTS_PATH)
    print("Model loaded successfully!")
class Grid(BaseModel):
    input_ids: List[int]
    bbox_subword_list: List[List[float]]
    texts: List[str]
    bbox_texts_list: List[List[float]]
# Add this model to define the expected structure
class GridDicts(BaseModel):
    grid_dicts: List[Grid]

from pydantic import BaseModel
from typing import List, Tuple

class Instance(BaseModel):
    boxes: List[List[float]]
    scores: List[float]
    classes: List[int]
    image_size: Tuple[int, int]

class SerializablePrediction(BaseModel):
    instances: Instance
    
def get_reading_order(predictions: List[SerializablePrediction]) -> List[SerializablePrediction]:
    """
    Sort predictions based on reading order (top-to-bottom, left-to-right with special handling for headers/footers)
    """
    # Define the type mapping based on the provided context
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
        # Map class IDs to segment types using the integrated type mapping
        return type_mapping.get(class_id, "Text")

    def sort_segments_by_position(segments_data: List[tuple]) -> List[tuple]:
        # Sort segments primarily by vertical position (top to bottom)
        # For segments close in vertical position, sort by horizontal position (left to right)
        vertical_threshold = 50  # Adjust this value based on your needs
        
        def sort_key(segment):
            box, _, _ = segment
            y = box[1]  # top coordinate
            x = box[0]  # left coordinate
            return (y // vertical_threshold, x)
        
        return sorted(segments_data, key=sort_key)

    updated_predictions = []
    
    for pred in predictions:
        # Create lists of segments grouped by type
        header_segments = []
        main_segments = []
        footer_segments = []
        
        # Group segments by type
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
        
        # Sort each group
        header_segments = sort_segments_by_position(header_segments)
        main_segments = sort_segments_by_position(main_segments)
        footer_segments = sort_segments_by_position(footer_segments)
        
        # Combine all segments in reading order
        ordered_segments = header_segments + main_segments + footer_segments
        
        # Create new prediction with ordered segments
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
    """
    Merge overlapping predictions, similar to the PDF segments implementation.
    """
    # Filter out low confidence predictions
    valid_indices = [i for i, score in enumerate(scores) if score >= 0.2]  # 0.2 corresponds to 20%
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
                # Calculate intersection
                intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                             max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                min_area = min(area1, area2)
                
                if intersection / min_area > 0.1:  # 10% overlap threshold
                    to_merge_indices.append(i)
            
            if to_merge_indices:
                merged = True
                merge_boxes = [box1] + [filtered_boxes[i] for i in to_merge_indices]
                merge_scores = [score1] + [filtered_scores[i] for i in to_merge_indices]
                merge_classes = [class1] + [filtered_classes[i] for i in to_merge_indices]
                
                # Remove merged boxes from original lists
                for i in reversed(to_merge_indices):
                    filtered_boxes.pop(i)
                    filtered_scores.pop(i)
                    filtered_classes.pop(i)
                
                # Merge boxes
                x1 = min(box[0] for box in merge_boxes)
                y1 = min(box[1] for box in merge_boxes)
                x2 = max(box[2] for box in merge_boxes)
                y2 = max(box[3] for box in merge_boxes)
                
                # Special handling for tables (class 9)
                if 8 in merge_classes:
                    final_class = 8
                else:
                    # Take the class of highest scoring prediction
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
    """
    Find the best segments by matching predictions with grid text elements.
    
    Args:
        predictions (List[SerializablePrediction]): Model predictions for each image
        grid_dicts (GridDicts): Text and bounding box information
        
    Returns:
        List[SerializablePrediction]: Updated predictions with best matches
    """
    grid_list = grid_dicts.grid_dicts if isinstance(grid_dicts, GridDicts) else grid_dicts
    
    updated_predictions = []
    for pred, grid in zip(predictions, grid_list):
        # First, merge overlapping predictions
        boxes, scores, classes = merge_colliding_predictions(
            pred.instances.boxes,
            pred.instances.scores,
            pred.instances.classes
        )
        
        # Create new lists for the text-based predictions
        text_boxes = []
        text_scores = []
        text_classes = []
        
        # Process each text element
        for text, bbox in zip(grid.texts, grid.bbox_texts_list):
            # Convert [x, y, w, h] to [x1, y1, x2, y2] format
            x, y, w, h = bbox
            text_box = [x, y, x + w, y + h]
            
            # Find best matching prediction
            best_score = 0
            best_class = 9  # Default to "Text"
            
            for box, score, cls in zip(boxes, scores, classes):
                # Calculate intersection
                intersection = max(0, min(text_box[2], box[2]) - max(text_box[0], box[0])) * \
                             max(0, min(text_box[3], box[3]) - max(text_box[1], box[1]))
                
                # Calculate areas
                text_area = (text_box[2] - text_box[0]) * (text_box[3] - text_box[1])
                pred_area = (box[2] - box[0]) * (box[3] - box[1])
                min_area = min(text_area, pred_area)
                
                # Use intersection over minimum area
                if min_area > 0 and intersection / min_area > 0.1 and score > best_score:
                    best_score = score
                    best_class = cls
            
            # Add the text element with its best matching prediction
            text_boxes.append(text_box)
            text_scores.append(best_score)
            text_classes.append(best_class)
        
        # Include predictions without associated text (e.g., images, tables)
        for box, score, cls in zip(boxes, scores, classes):
            has_overlap = False
            for text_box in text_boxes:
                intersection = max(0, min(text_box[2], box[2]) - max(text_box[0], box[0])) * \
                             max(0, min(text_box[3], box[3]) - max(text_box[1], box[1]))
                if intersection > 0:
                    has_overlap = True
                    break
            
            if not has_overlap:
                text_boxes.append(box)
                text_scores.append(score)
                text_classes.append(cls)
        
        # Create updated prediction
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


@app.post("/batch/")
async def process_image_batch_endpoint(
    files: List[UploadFile] = File(...),
    grid_dicts: str = Form(...)
):    
    # Parse the JSON string into GridDicts
    grid_dicts_data = json.loads(grid_dicts)
    grid_dicts_obj = GridDicts(grid_dicts=grid_dicts_data)

    
    # Read all images
    images = []
    for file in files:
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        images.append(image)
    
    # Process the batch
    raw_predictions, _ = process_image_batch(predictor, images, dataset_name="doclaynet", grid_dicts=grid_dicts_data)
    
    # Convert raw predictions to SerializablePrediction format
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
    
    # Find best segments
    final_predictions = find_best_segments(predictions, grid_dicts_obj)
    ordered_predictions = get_reading_order(final_predictions)
    
    return ordered_predictions
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)