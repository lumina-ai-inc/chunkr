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
 
            
        # Convert image to RGB if it's not already
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')
        
        # Get table structure
        cells = get_table_structure(image)
        # Annotate and save cells on image as raw_annotate_output
        from PIL import ImageDraw

        # Create a copy of the image for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Draw bounding boxes for each detected cell
        for cell in cells:
            bbox = cell['bbox']
            draw.rectangle(bbox, outline='red', width=2)

        # Save the annotated image
        output_path = 'raw_annotate_output.png'
        annotated_image.save(output_path)
        logging.info(f"Raw annotated image saved as {output_path}")
        # Get cell coordinates
        cell_coordinates = get_cell_coordinates_by_row(cells,merge_threshold=0.14, raw_output=False)
        
        # Convert cell_coordinates (list of Row objects) to JSON-serializable format
        serializable_cell_coordinates = [row.to_dict() for row in cell_coordinates]
        
        return image, serializable_cell_coordinates
        # return image,cells
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise

async def process_table_image(file):
    _, cell_coordinates = await process_image(file)
    return cell_coordinates

