import os
import torch
import uuid
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
import gc
from pathlib import Path
from dotenv import load_dotenv
import logging
import traceback
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException  
from typing import List, Tuple, Optional

from models import BoundingBox, BoundingBoxOutput, InstanceOutput, Instance, SerializablePrediction, FinalPrediction
from prediction import map_yolo_to_segment_type, get_reading_order_and_merge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / '.env'

load_dotenv(dotenv_path=ENV_PATH)

# Configuration from environment variables
batch_wait_time = float(os.getenv("BATCH_WAIT_TIME", 0.1))
max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 10))
overlap_threshold = float(os.getenv("OVERLAP_THRESHOLD", 0.1))
score_threshold = float(os.getenv("SCORE_THRESHOLD", 0.15))
conf_threshold = float(os.getenv("CONF_THRESHOLD", 0.1))
imgsz = int(os.getenv("IMAGE_SIZE", 1024))

print(f"Max batch size: {max_batch_size}")
print(f"Overlap threshold: {overlap_threshold}")
print(f"Score threshold: {score_threshold}")
print(f"Confidence threshold: {conf_threshold}")
print(f"Image size: {imgsz}")
print(f"Batch wait time: {batch_wait_time}")


# Global model variable
model = None

def download_model():
    filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
    model = YOLOv10(filepath)
    print("Model loaded successfully!")
    return model

@asynccontextmanager    
async def lifespan(app: FastAPI):
    global model
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = download_model()
    yield
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_detail = str(exc)
    logger.error(f"Request validation error: {error_detail}")
    logger.error(f"Request body: {await request.body()}")
    logger.error(f"Request headers: {request.headers}")
    return JSONResponse(
        status_code=422,
        content={"detail": f"Validation error: {error_detail}"}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {str(exc)}"}
    )

@app.post("/batch")
async def batch_od(
    files: List[UploadFile] = File(...),
):
    try:
        logger.info(f"Received batch request with {len(files)} files")
        results = []
        image_data_list = []

        # Read all files first
        for i, file in enumerate(files):
            try:
                logger.info(f"Reading file {i+1}/{len(files)}: {file.filename}")
                image_data = await file.read()
                logger.info(f"File {file.filename} size: {len(image_data)} bytes")
                image_data_list.append(image_data)
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {str(e)}")
                logger.error(traceback.format_exc())
                # Inform client about the specific file error but continue reading others
                raise HTTPException(status_code=500, detail=f"Error reading file {file.filename}: {str(e)}")

        # Process the batch if any images were successfully read
        if image_data_list:
            logger.info(f"Processing batch of {len(image_data_list)} images.")
            # Call the batch processing function
            _, final_preds = await process_image_batch(
                image_data_list,
                conf=conf_threshold,
                img_size=imgsz # Pass global config or handle defaults within function
            )
            results.extend(final_preds)
        else:
            logger.warning("No image data collected for batch processing.")

        return results
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (e.g., from file reading) to ensure proper client response
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during the batch process
        logger.error(f"Batch processing overall failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "YOLO DocLayout API"}


async def process_image(image_data: bytes, conf: float = None, img_size: int = None):
    if conf is None:
        conf = conf_threshold
    if img_size is None:
        img_size = imgsz
        
    temp_file = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_file, "wb") as f:
        f.write(image_data)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        det_res = model.predict(
            temp_file,
            imgsz=img_size,
            conf=conf,
            device=device,
        )
        
        if det_res[0].boxes is not None:
            boxes = det_res[0].boxes.xyxy.cpu().numpy()
            cls = det_res[0].boxes.cls.cpu().numpy().astype(int)
            conf = det_res[0].boxes.conf.cpu().numpy()
            
            # Get image height for class mapping
            img_height = det_res[0].orig_shape[0]
            
            # Map YOLO classes to SegmentType
            mapped_classes = []
            found_title = False  # Track if we've already found a title
            for i, yolo_class in enumerate(cls):
                box_y = boxes[i, 1]  # Y coordinate for position-based mapping
                is_first_title = False if found_title and yolo_class == 0 else True
                mapped_class = map_yolo_to_segment_type(yolo_class, box_y, img_height, is_first_title)
                
                # Update found_title flag if this was a title
                if yolo_class == 0 and not found_title:
                    found_title = True
                    
                mapped_classes.append(mapped_class)
            
            # Convert boxes to BoundingBox instances
            bbox_list = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]) for box in boxes]

            # Apply reading order and merge overlapping boxes
            ordered_boxes, ordered_scores, ordered_classes, image_size = get_reading_order_and_merge(
                bbox_list, conf.tolist(), mapped_classes, det_res[0].orig_shape, score_threshold, overlap_threshold
            )
            
            # Create the output format
            bbox_output_list = [BoundingBoxOutput(
                left=box.x1,
                top=box.y1,
                width=box.x2 - box.x1,
                height=box.y2 - box.y1
            ) for box in ordered_boxes]
            
            # Create instance objects with mapped classes
            instance = Instance(
                boxes=ordered_boxes,
                scores=ordered_scores,
                classes=ordered_classes,
                image_size=image_size
            )
            
            instance_output = InstanceOutput(
                boxes=bbox_output_list,
                scores=ordered_scores,
                classes=ordered_classes,
                image_size=image_size
            )
            
            serializable_pred = SerializablePrediction(instances=instance)
            final_pred = FinalPrediction(instances=instance_output)
            
            return serializable_pred, final_pred
        else:
            # Empty results
            return SerializablePrediction(
                instances=Instance(boxes=[], scores=[], classes=[], image_size=det_res[0].orig_shape)
            ), FinalPrediction(
                instances=InstanceOutput(boxes=[], scores=[], classes=[], image_size=det_res[0].orig_shape)
            )
    
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

# Added function for batch processing
async def process_image_batch(
    image_data_list: List[bytes],
    conf: Optional[float] = None,
    img_size: Optional[int] = None
) -> Tuple[List[SerializablePrediction], List[FinalPrediction]]:

    # Use global defaults if specific values not provided
    if conf is None:
        conf = conf_threshold
    if img_size is None:
        img_size = imgsz

    temp_files = []
    serializable_preds = []
    final_preds = []

    # Prepare temporary files for batch processing
    for image_data in image_data_list:
        temp_file = f"temp_{uuid.uuid4()}.jpg"
        try:
            with open(temp_file, "wb") as f:
                f.write(image_data)
            temp_files.append(temp_file)
        except Exception as e:
            logger.error(f"Failed to write temporary file: {str(e)}")
            # Clean up any files created so far before raising
            for f in temp_files:
                if os.path.exists(f):
                    try: os.remove(f) # Add cleanup for created files
                    except Exception as cleanup_e: logger.error(f"Failed to remove temp file during error handling: {cleanup_e}")
            raise HTTPException(status_code=500, detail=f"Failed to create temporary file for processing: {str(e)}")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        logger.info(f"Using device: {device} for batch prediction on {len(temp_files)} images")

        if not temp_files:
             logger.warning("No temporary files created for batch prediction.")
             return [], [] # Return empty lists if no files to process

        # Call the predict method with a list of temp files for batch processing
        det_res = model.predict(
            temp_files,  # Pass the list of temp files
            imgsz=img_size,
            conf=conf,
            device=device,
            # verbose=False # Optionally reduce verbosity if needed
        )
        logger.info(f"Model prediction completed for {len(det_res)} images.")

        # Process results for each image in the batch
        for i, res in enumerate(det_res):
            img_shape = res.orig_shape if hasattr(res, 'orig_shape') else (0, 0) # Get shape early

            # Ensure that 'res' has the expected structure before accessing attributes
            if hasattr(res, 'boxes') and res.boxes is not None and hasattr(res.boxes, 'xyxy'):
                boxes = res.boxes.xyxy.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                conf_scores = res.boxes.conf.cpu().numpy()

                if boxes.shape[0] == 0: # Check if any boxes were detected
                    logger.info(f"No boxes detected for image {i} in batch (shape: {img_shape}). Creating empty prediction.")
                    serializable_preds.append(SerializablePrediction(instances=Instance(boxes=[], scores=[], classes=[], image_size=img_shape)))
                    final_preds.append(FinalPrediction(instances=InstanceOutput(boxes=[], scores=[], classes=[], image_size=img_shape)))
                    continue # Skip to next image

                img_height = img_shape[0]

                # Map YOLO classes to SegmentType
                mapped_classes = []
                found_title = False
                for j, yolo_class in enumerate(cls):
                    box_y = boxes[j, 1]
                    is_first_title = not found_title and yolo_class == 0
                    mapped_class = map_yolo_to_segment_type(yolo_class, box_y, img_height, is_first_title)
                    if is_first_title:
                        found_title = True
                    mapped_classes.append(mapped_class)

                # Convert boxes to BoundingBox instances
                bbox_list = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]) for box in boxes]

                # Apply reading order and merge overlapping boxes
                ordered_boxes, ordered_scores, ordered_classes, image_size = get_reading_order_and_merge(
                    bbox_list, conf_scores.tolist(), mapped_classes, img_shape, score_threshold, overlap_threshold
                )

                # Create the output format
                bbox_output_list = [BoundingBoxOutput(
                    left=box.x1,
                    top=box.y1,
                    width=box.x2 - box.x1,
                    height=box.y2 - box.y1
                ) for box in ordered_boxes]

                # Create instance objects with mapped classes
                instance = Instance(
                    boxes=ordered_boxes,
                    scores=ordered_scores,
                    classes=ordered_classes,
                    image_size=image_size
                )

                instance_output = InstanceOutput(
                    boxes=bbox_output_list,
                    scores=ordered_scores,
                    classes=ordered_classes,
                    image_size=image_size
                )

                serializable_pred = SerializablePrediction(instances=instance)
                final_pred = FinalPrediction(instances=instance_output)

                serializable_preds.append(serializable_pred)
                final_preds.append(final_pred)
            else:
                # Handle cases where 'res' or 'res.boxes' is None or lacks expected attributes
                logger.warning(f"No boxes detected or result format unexpected for image {i} in batch (shape: {img_shape}). Creating empty prediction.")
                serializable_preds.append(SerializablePrediction(instances=Instance(boxes=[], scores=[], classes=[], image_size=img_shape)))
                final_preds.append(FinalPrediction(instances=InstanceOutput(boxes=[], scores=[], classes=[], image_size=img_shape)))

        return serializable_preds, final_preds

    except Exception as e:
        logger.error(f"Error during batch model prediction or processing: {str(e)}")
        logger.error(traceback.format_exc())
        # Ensure cleanup happens before raising
        raise HTTPException(status_code=500, detail=f"Error during batch model prediction: {str(e)}")
    finally:
        # Clean up temporary files
        logger.info(f"Cleaning up {len(temp_files)} temporary files.")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Failed to remove temporary file {temp_file}: {str(e)}")

        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)