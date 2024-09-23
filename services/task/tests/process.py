from PIL import Image, ImageDraw
import requests
import os
import json
import time
from pathlib import Path
import glob
import uuid
import base64
import io 
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_base_segments_list(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    base_segments = []
    for chunk in json_data:
        if 'segments' in chunk:
            for base_segment in chunk['segments']:
                base_segments.append({
                    'segment_id': str(uuid.uuid4()),
                    'left': base_segment['left'],
                    'top': base_segment['top'],
                    'width': base_segment['width'],
                    'height': base_segment['height'],
                    'page_number': base_segment['page_number'],
                    'page_width': base_segment['page_width'],
                    'page_height': base_segment['page_height'],
                    'text': base_segment['text'],
                    'segment_type': base_segment['type'],
                })
    
    return base_segments


def send_files_to_process(pdf_path: str, json_path: str, service_url: str, output_dir: str) -> dict:
    """
    Send a PDF file and JSON file to the OCR service and return the results.

    :param pdf_path: Path to the PDF file
    :param json_path: Path to the JSON file containing segments
    :param service_url: URL of the OCR service
    :param output_dir: Directory to save output files
    :return: Dictionary containing OCR results and time taken
    """
    base_segments = create_base_segments_list(json_path)

    files = {
        'file': open(pdf_path, 'rb'),
    }

    data = {
        'base_segments': json.dumps(base_segments),
        'page_image_density': 300,
        'page_image_extension': 'jpg',
        'segment_image_extension': 'jpg',
        'segment_image_density': 300,
        'segment_bbox_offset': 2,
        'segment_image_quality': 100,
        'segment_image_resize': None,
        'num_workers': None,
        'ocr_strategy': 'Auto'
    }

    start_time = time.time()
    response = requests.post(f"{service_url}/process", files=files, data=data, timeout=600)
    time_taken = time.time() - start_time

    print(f"Time taken for processing: {time_taken:.2f} seconds")

    if response.status_code == 200:
        results = response.json()
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # Save JSON result
        json_path = os.path.join(output_dir, f"{base_name}_result.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        for index, segment in enumerate(results):
            if 'image' in segment and segment['image']:
                image_data = base64.b64decode(segment['image'])
                image = Image.open(io.BytesIO(image_data))
                try:
                    draw = ImageDraw.Draw(image)
                    if 'ocr' in segment:
                        for ocr_result in segment['ocr']:
                            bbox = ocr_result['bbox']
                            draw.rectangle([
                                (bbox['top_left'][0], bbox['top_left'][1]),
                                (bbox['bottom_right'][0], bbox['bottom_right'][1])
                            ], outline="red", width=1)
                except Exception as e:
                    print(f"Error drawing bbox: {e}")
                image_path = os.path.join(output_dir, f"{index}.jpg")
                image.save(image_path)
            
        return results
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


def process_pdf_and_json(pdf_path: str, json_path: str, service_url: str, output_dir: str) -> dict:
    """
    Process a PDF file and its corresponding JSON file with segments.

    :param pdf_path: Path to the PDF file
    :param json_path: Path to the JSON file containing segments
    :param service_url: URL of the OCR service
    :param output_dir: Directory to save output files
    :return: Dictionary containing OCR results
    """
    os.makedirs(output_dir, exist_ok=True)
    return send_files_to_process(pdf_path, json_path, service_url, output_dir)


if __name__ == "__main__":
    service_url = os.getenv('SERVICE_URL')
    if not service_url:
        raise ValueError("SERVICE_URL not found in environment variables")

    run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    input_dir = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/process/2403.12313"
    output_dir = f"/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/output/process/{run}/2403.12313"

    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    if not pdf_files:
        print("No PDF file found in the input directory.")
        exit(1)

    pdf_path = pdf_files[0]
    base_name = Path(pdf_path).stem
    json_path = os.path.join(input_dir, f"{base_name}.json")

    if not os.path.exists(json_path):
        print(f"No matching JSON file found for {base_name}")
        exit(1)

    start_time = time.time()
    print(f"Processing {base_name}")
    results = process_pdf_and_json(pdf_path, json_path, service_url, output_dir)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")