import io
from PIL import Image
from table_structure import get_table_structure
from utils import  get_cell_coordinates_by_row
import logging
import cv2
import numpy as np

def preprocess_image(image):
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Convert back to PIL Image
    preprocessed_image = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    
    return preprocessed_image


async def process_image(file, preprocess=True):

    try:
        # Check if file is already bytes
        if isinstance(file, bytes):
            image_content = file
        else:
            # Assume it's a file-like object
            image_content = await file.read()
        # Open the image
        image = Image.open(io.BytesIO(image_content))
        
        # Preprocess the image
        if preprocess:
            image = preprocess_image(image)
        
        # Get table structure
        cells = get_table_structure(image)
        # Annotate and save cells on image as raw_annotate_output
        
        # Get cell coordinates
        cell_coordinates = get_cell_coordinates_by_row(cells,merge_threshold=0.14, raw_output=False)
        
        serializable_cell_coordinates = [row.to_dict() for row in cell_coordinates]
        
        return image, serializable_cell_coordinates
        # return image,cells
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

async def process_table_image(file):
    _, cell_coordinates = await process_image(file)
    return cell_coordinates

