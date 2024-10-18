import io
from PIL import Image
from table_structure import get_table_structure
from utils import  get_cell_coordinates_by_row
import logging
import cv2
import numpy as np


async def process_image(file, preprocess=False):

    try:
        # Check if file is already bytes
        if isinstance(file, bytes):
            image_content = file
        else:
            # Assume it's a file-like object
            image_content = await file.read()
        
        image = Image.open(io.BytesIO(image_content))
 
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

