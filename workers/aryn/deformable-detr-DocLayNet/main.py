from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch
from PIL import Image
from typing import List
import io

app = FastAPI()

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Aryn/deformable-detr-DocLayNet")
model = DeformableDetrForObjectDetection.from_pretrained("Aryn/deformable-detr-DocLayNet")

class Detection(BaseModel):
    label: str
    confidence: float
    box: List[float]

@app.post("/detect", response_model=List[Detection])
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process the results
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]
        
        # Format the results
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append(Detection(
                label=model.config.id2label[label.item()],
                confidence=round(score.item(), 3),
                box=[round(i, 2) for i in box.tolist()]
            ))

        return detections

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)