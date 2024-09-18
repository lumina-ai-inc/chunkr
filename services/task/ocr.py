from paddleocr import PaddleOCR
from pathlib import Path
from PIL import Image
import numpy as np

def perform_paddle_ocr(ocr: PaddleOCR, image_path: Path) -> list:
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img_array = np.array(img)
        result = ocr.ocr(img_array)
    except Image.DecompressionBombError:
        print(f"Error: The image file is too large or complex to process.")
        result = []
    except Image.UnidentifiedImageError:
        print(f"Error: The image file '{image_path}' cannot be identified or is corrupted.")
        result = []
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        result = []
    return result