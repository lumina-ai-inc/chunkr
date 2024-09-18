from paddleocr import PaddleOCR
from pathlib import Path
from PIL import Image
import numpy as np

def perform_paddle_ocr(ocr: PaddleOCR, image_path: Path) -> list:
    try:
        # Open the image using PIL and convert to RGB
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            # Convert PIL Image to numpy array
            img_array = np.array(img)
        
        # Pass the numpy array to PaddleOCR
        result = ocr.ocr(img_array)
    except Exception as e:
        result = []
        print(f"An error occurred: {str(e)}")
    return result