import requests
from config import get_tesseract_url
import easyocr
import json
from model_config import EASYOCR_LOCAL_MODEL_PATH

# Initialize EasyOCR with a fallback mechanism
try:
    reader = easyocr.Reader(['en'], model_storage_directory=EASYOCR_LOCAL_MODEL_PATH, download_enabled=True)
except AttributeError:
    # Fallback to CPU if MPS is not available
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=EASYOCR_LOCAL_MODEL_PATH, download_enabled=True)

def ocr_tesseract(image):
    tesseract_url = get_tesseract_url()
    
    # Prepare the options for Tesseract
    options = {
        "languages": ["eng"]
    }
    
    # Prepare the files for the request
    files = {
        "file": ("image.png", image, "image/png"),
        "options": (None, json.dumps(options))
    }
    
    # Make the POST request to the Tesseract API
    response = requests.post(f"{tesseract_url}/tesseract", files=files)
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        data = result["data"]
        if data is None:
            raise Exception(f"OCR failed")
        if data["stdout"] == "":
            if data['stderr'] == "":
                raise Exception(f"OCR failed with error: {result['data']['stderr']}")
        return data["stdout"]
    else:
        raise Exception(f"OCR request failed with status code {response.status_code}")


def ocr_easyocr(image):
    result = reader.readtext(image)
    return result
