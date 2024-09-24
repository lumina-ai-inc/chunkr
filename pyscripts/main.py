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

from api import process_file
from download import download_file
from models import Model, TableOcr
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

def extract_and_annotate_file(file_path: str, model: Model, table_ocr: TableOcr = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(file_path).split(".")[0]
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_json_path = os.path.join(output_dir, f"{file_name}_json.json")
    output_annotated_path = os.path.join(output_dir, f"{file_name}_annotated.pdf")

    print(f"Processing file: {file_path}")
    task = process_file(file_path, model, table_ocr)
    output = task.output
    print(f"File processed: {file_path}")

    if output is None:
        raise Exception(f"Output not found for {file_path}")

    print(f"Downloading bounding boxes for {file_path}...")
    output_json_path = save_to_json(output_json_path, output, file_name)
    print(f"Downloaded bounding boxes for {file_path}")

    print(f"Annotating file: {file_path}")
    draw_bounding_boxes(file_path, output, output_annotated_path)
    print(f"File annotated: {file_path}")

def throughput_test(growth_func: GrowthFunc, start_page: int, end_page: int, num_pdfs: int, model: Model, table_ocr: TableOcr = None):
    print("Starting throughput test...")
    if not isinstance(growth_func, GrowthFunc):
        raise ValueError("growth_func must be an instance of GrowthFunc Enum")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input")
    run_id = f"run_{uuid.uuid4().hex}"
    run_dir = os.path.join(input_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")

    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    if not pdf_files:
        raise ValueError("No PDF files found in the input folder.")
    original_pdf = pdf_files[0]
    pdf_test_paths = []

    # Create a new PDF with only the specified page range
    base_pdf_writer = PdfWriter()
    with open(original_pdf, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(start_page - 1, end_page):
            base_pdf_writer.add_page(pdf_reader.pages[page_num])

    base_pdf_path = os.path.join(run_dir, f"{os.path.basename(original_pdf).split('.')[0]}_base.pdf")
    with open(base_pdf_path, 'wb') as output_file:
        base_pdf_writer.write(output_file)

    # Create all test PDFs
    for i in range(num_pdfs):
        if growth_func == GrowthFunc.LINEAR:
            multiplier = i + 1
        elif growth_func == GrowthFunc.EXPONENTIAL:
            multiplier = 2 ** i
        elif growth_func == GrowthFunc.LOGARITHMIC:
            multiplier = max(1, int(np.log2(i + 2)))
        elif growth_func == GrowthFunc.QUADRATIC:
            multiplier = (i + 1) ** 2
        elif growth_func == GrowthFunc.CUBIC:
            multiplier = (i + 1) ** 3
        else:
            raise ValueError("Unsupported growth function")

        test_pdf_name = f"{os.path.basename(original_pdf).split('.')[0]}_{multiplier}x.pdf"
        test_pdf_path = os.path.join(run_dir, test_pdf_name)

        # Create the test PDF by duplicating the base PDF
        test_pdf_writer = PdfWriter()
        for _ in range(multiplier):
            with open(base_pdf_path, 'rb') as base_file:
                base_reader = PdfReader(base_file)
                for page in base_reader.pages:
                    test_pdf_writer.add_page(page)

        with open(test_pdf_path, 'wb') as output_file:
            test_pdf_writer.write(output_file)

        print(f"Created {test_pdf_path} with multiplier {multiplier}x")
        pdf_test_paths.append(test_pdf_path)

    # Process all created PDFs and log results
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"throughput_results_{growth_func.value}_{run_id}.csv")
    fieldnames = ['PDF Name', 'Number of Pages', 'Processing Time (seconds)', 'Throughput (pages/sec)']
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        for pdf_path in pdf_test_paths:
            start_time = time.time()
            try:
                extract_and_annotate_file(pdf_path, model, table_ocr)
            except Exception as e:
                print(f"Failed to process {pdf_path}: {str(e)}")
                continue
            end_time = time.time()
            
            processing_time = end_time - start_time
            num_pages = len(PdfReader(pdf_path).pages)
            throughput = num_pages / processing_time
            
            row_data = {
                'PDF Name': os.path.basename(pdf_path),
                'Number of Pages': num_pages,
                'Processing Time (seconds)': processing_time,
                'Throughput (pages/sec)': throughput
            }
            
            writer.writerow(row_data)
            csvfile.flush()  # Ensure data is written to file
            print(f"Processed {os.path.basename(pdf_path)}: {num_pages} pages in {processing_time:.2f} seconds. Throughput: {throughput:.2f} pages/sec")

    print(f"Throughput test results saved to {csv_path}")
    print("Throughput test completed successfully.")

if __name__ == "__main__":
    model = Model.HighQuality
    table_ocr = None
    throughput_test(GrowthFunc.LINEAR, start_page=1, end_page=40, num_pdfs=10, model=model, table_ocr=table_ocr)
    print("Throughput test completed.")
