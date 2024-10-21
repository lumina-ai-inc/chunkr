import os
from datetime import datetime
import concurrent.futures
from functools import partial
import glob
import time
import csv
import uuid
from enum import Enum
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
import requests
import urllib.request
from api import process_file
from download import download_file
from models import Model, TableOcr, OcrStrategy, UploadForm, TaskResponse
from annotate import draw_bounding_boxes

import json

class GrowthFunc(Enum):
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'
    LOGARITHMIC = 'logarithmic'
    QUADRATIC = 'quadratic'
    CUBIC = 'cubic'

def print_time_taken(created_at, finished_at):
    if created_at and finished_at:
        try:
            start_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            end_time = datetime.fromisoformat(
                finished_at.strip('"').replace(" UTC", "+00:00")
            )
            time_taken = end_time - start_time
            print(f"Time taken: {time_taken}")
        except ValueError:
            print("Unable to calculate time taken due to invalid timestamp format")
    else:
        print("Time taken information not available")

def save_to_json(file_path: str, output: json, file_name: str ):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{file_name}_json.json")
    with open(output_json_path, "w") as f:
        json.dump(output, f)
    return output_json_path

def extract_and_annotate_file(file_path: str, model: Model, target_chunk_length: int = None, ocr_strategy: OcrStrategy = OcrStrategy.Auto):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(file_path).split(".")[0]
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_json_path = os.path.join(output_dir, f"{file_name}_json.json")
    output_annotated_path = os.path.join(output_dir, f"{file_name}_annotated.pdf")

    print(f"Processing file: {file_path}")
    upload_form = UploadForm(file=file_path, model=model, target_chunk_length=target_chunk_length, ocr_strategy=ocr_strategy)
    task: TaskResponse = process_file(upload_form)
    output = task.output
    print(f"File processed: {file_path}")

    if output is None:
        raise Exception(f"Output not found for {file_path}")

    print(f"Downloading bounding boxes for {file_path}...")
    output_json_path = save_to_json(output_json_path, output, file_name)
    print(f"Downloaded bounding boxes for {file_path}")

    if task.pdf_location:
        temp_pdf_path = os.path.join(output_dir, f"{file_name}_temp.pdf")
        urllib.request.urlretrieve(task.pdf_location, temp_pdf_path)
        print(f"Annotating file: {temp_pdf_path}")
        draw_bounding_boxes(temp_pdf_path, output, output_annotated_path)
        os.remove(temp_pdf_path)
    else:
        draw_bounding_boxes(file_path, output, output_annotated_path)
    print(f"File annotated: {file_path}")

import concurrent.futures
import glob

def main(max_workers: int = None, model: Model = Model.HighQuality, target_chunk_length: int = None, ocr_strategy: OcrStrategy = OcrStrategy.Auto, dir="input"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir,dir)
    input_files = []
    for extension in ["*.pdf", "*.docx", "*.ppt"]:
        input_files.extend(glob.glob(os.path.join(input_dir, extension)))

    if not input_files:
        print("No PDF files found in the input folder.")
        return

    print(f"Processing {len(input_files)} files with {max_workers} parallel workers...")
    elapsed_times = []

    def timed_extract(file_path):
        start_time = time.time()
        extract_and_annotate_file(file_path, model, target_chunk_length, ocr_strategy)
        end_time = time.time()
        return {'file_path': file_path, 'elapsed_time': end_time - start_time}
        
    max_workers = max_workers or len(input_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(timed_extract, file_path) for file_path in input_files]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                elapsed_times.append(result)
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    print("All files processed.")
    return elapsed_times




if __name__ == "__main__":
    model = Model.Fast
    target_chunk_length = 1000
    ocr_strategy = OcrStrategy.Auto
    times = main(None, model, target_chunk_length, ocr_strategy, "input")
    
    total_time = sum(result['elapsed_time'] for result in times)
    print(f"Total time taken to process all files: {total_time:.2f} seconds")
    
    for result in times:
        print(f"Time taken to process {result['file_path']}: {result['elapsed_time']:.2f} seconds")


