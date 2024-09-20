import requests
import os
import json
import time
from typing import List
from pathlib import Path
import glob
import uuid

def create_segments_list(json_data):
    segments = []
    
    for item in json_data:
        if 'segments' in item:
            for segment in item['segments']:
                segments.append({
                    'left': segment['left'],
                    'top': segment['top'],
                    'width': segment['width'],
                    'height': segment['height'],
                    'page_number': segment['page_number'],
                    'page_width': segment['page_width'],
                    'page_height': segment['page_height'],
                    'text': segment['text'],
                    'type': segment['type'],
                    'segment_id': uuid.uuid4(),
                    'ocr': None,
                    'image': None
                })
    
    return segments


def send_files_to_process(pdf_path: str, json_path: str, service_url: str, output_dir: str) -> dict:
    """
    Send a PDF file and JSON file to the OCR service and return the results.

    :param pdf_path: Path to the PDF file
    :param json_path: Path to the JSON file containing segments
    :param service_url: URL of the OCR service
    :param output_dir: Directory to save output files
    :return: Dictionary containing OCR results and time taken
    """
    segments = create_segments_list(json_path)

    files = {
        'file': open(pdf_path, 'rb'),
    }

    data = {
        'segments': json.dumps(segments),
        'image_density': 300,
        'page_image_extension': 'png',
        'segment_image_extension': 'jpg'
    }

    start_time = time.time()
    response = requests.post(f"{service_url}/process", files=files, data=data)
    time_taken = time.time() - start_time

    print(f"Time taken for processing: {time_taken:.2f} seconds")

    if response.status_code == 200:
        results = response.json()
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # Save JSON result
        json_path = os.path.join(output_dir, f"{base_name}_result.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

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
    service_url = "http://35.236.179.125:3000"
    input_dir = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/process/00c08086-9837-5551-8133-4e22ac28c6a5"
    output_dir = "/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/output/process/00c08086-9837-5551-8133-4e22ac28c6a5"

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