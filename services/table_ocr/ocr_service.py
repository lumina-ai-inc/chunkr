import io
from PIL import Image
from table_structure import get_table_structure
from utils import  get_cell_coordinates_by_row
import logging
import cv2
import numpy as np
from PIL import ImageEnhance, ImageFilter

async def preprocess_image(image: Image.Image) -> Image.Image:
    gray_image = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)
    denoised_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
    np_image = np.array(denoised_image)
    binary_image = cv2.adaptiveThreshold(
        np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(binary_image)
async def preprocess_image_path(image_path: str) -> Image.Image:
    image = Image.open(image_path)
    gray_image = image.convert('L')
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)
    denoised_image = enhanced_image.filter(ImageFilter.MedianFilter(size=3))
    np_image = np.array(denoised_image)
    binary_image = cv2.adaptiveThreshold(
        np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(binary_image)
async def process_image(file, preprocess=True):

    try:
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content))
        if preprocess:
            image = await preprocess_image(image)
            
        # Convert image to RGB if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get table structure
        cells = get_table_structure(image)
        
        # Get cell coordinates
        cell_coordinates = get_cell_coordinates_by_row(cells,merge_threshold=0.16, raw_output=False)
        
        return image, cell_coordinates
        # return image,cells
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

async def process_table_image(file):
    _, cell_coordinates = await process_image(file)
    return cell_coordinates

