import os
import torch
from transformers import TableTransformerForObjectDetection
from torchvision import transforms
from config import DEVICE, MAX_IMAGE_SIZE
from utils import MaxResize, outputs_to_objects
from model_config import TABLE_STRUCTURE_LOCAL_MODEL_PATH, TABLE_STRUCTURE_REMOTE_MODEL_NAME

def load_or_download_model():
    if os.path.exists(TABLE_STRUCTURE_LOCAL_MODEL_PATH):
        structure_model = TableTransformerForObjectDetection.from_pretrained(TABLE_STRUCTURE_LOCAL_MODEL_PATH)
    else:
        structure_model = TableTransformerForObjectDetection.from_pretrained(TABLE_STRUCTURE_REMOTE_MODEL_NAME)
        structure_model.save_pretrained(TABLE_STRUCTURE_LOCAL_MODEL_PATH)
    
    return structure_model.to(DEVICE)

structure_model = load_or_download_model()

structure_transform = transforms.Compose([
    MaxResize(MAX_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_table_structure(image):
    pixel_values = structure_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = structure_model(pixel_values)

    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, image.size, structure_id2label)
    return cells