import os
from doctr.models import ocr_predictor

if __name__ == "__main__":
    detection_model = os.environ.get("OCR_DETECTION_MODEL", "db_resnet50")
    recognition_model = os.environ.get("OCR_RECOGNITION_MODEL", "parseq")
    
    predictor = ocr_predictor(detection_model, recognition_model, pretrained=True, 
                             export_as_straight_boxes=True)
