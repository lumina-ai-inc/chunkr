import io
from PIL import Image
from table_structure import get_table_structure
from utils import apply_ocr, get_cell_coordinates_by_row
import logging

async def process_image(file):
    try:
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content))
        
        # Convert image to RGB if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get table structure
        cells = get_table_structure(image)
        
        # Get cell coordinates
        cell_coordinates = get_cell_coordinates_by_row(cells)
        
        return image, cell_coordinates
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

async def process_table_image(file):
    _, cell_coordinates = await process_image(file)
    return cell_coordinates

async def ocr_table_image(file, ocr_model):

    image, cell_coordinates = await process_image(file)
    
    # Apply OCR
    data = apply_ocr(cell_coordinates, image, ocr_model)
    
    return data