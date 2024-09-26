
import unittest
import requests
import os
import json
from pathlib import Path
from annotate import draw_bounding_boxes
import dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import uuid
import time
import numpy as np
import glob
import shutil
import csv
from PyPDF2 import PdfReader, PdfWriter
import concurrent.futures

dotenv.load_dotenv(override=True)

class GrowthFunc(Enum):
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'
    LOGARITHMIC = 'logarithmic'
    QUADRATIC = 'quadratic'
    CUBIC = 'cubic'

class TestPDLAServer(unittest.TestCase):
    BASE_URL = os.getenv("PDLASERVER_URL")
    MAX_WORKERS = 4  

    def setUp(self):
        print("Setting up test environment...")
        self.input_folder = Path(__file__).parent / "input"
        self.output_folder = Path(__file__).parent / "output"
        if not self.input_folder.exists() or not self.input_folder.is_dir():
            raise FileNotFoundError(f"Input folder not found at {self.input_folder}")
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True)
        
        self.test_pdfs = list(self.input_folder.glob("*.pdf"))
        if not self.test_pdfs:
            raise FileNotFoundError(f"No PDF files found in {self.input_folder}")
        print(f"Found {len(self.test_pdfs)} PDF files for testing.")

    def call_fast_pdla(self, pdf_path):
        url = f"{self.BASE_URL}/analyze/fast"
        print(f"Processing fast PDF file: {pdf_path}")
        with open(pdf_path, "rb") as pdf_file:
            files = {"file": (pdf_path.name, pdf_file, "application/pdf")}
            response = requests.post(url, files=files)
        return response

    def call_high_quality_pdla(self, pdf_path):
        url = f"{self.BASE_URL}/analyze/high-quality"
        print(f"Processing high quality PDF file: {pdf_path}")
        with open(pdf_path, "rb") as pdf_file:
            files = {"file": (pdf_path.name, pdf_file, "application/pdf")}
            data = {"density": 72, "extension": "jpeg"}
            response = requests.post(url, files=files, data=data)
        return response

    def process_pdf(self, pdf_path, fast=True):
        if fast:
            response = self.call_fast_pdla(pdf_path)
        else:
            response = self.call_high_quality_pdla(pdf_path)
        
        print(f"Response status code for {pdf_path.name}: {response.status_code}")
        self.assertEqual(response.status_code, 200)
        
        json_response = response.json()
        print(f"Received JSON response with {len(json_response)} items for {pdf_path.name}.")
        self.assertIsInstance(json_response, list)
        self.assertTrue(len(json_response) > 0)
        
        # Check if the response contains expected fields
        first_item = json_response[0]
        expected_fields = ["left", "top", "width", "height", "page_number", "page_width", "page_height", "text", "type"]
        for field in expected_fields:
            print(f"Checking for field: {field} in {pdf_path.name}")
            self.assertIn(field, first_item)
        
        # Save output to JSON file
        output_json_path = self.output_folder / f"{pdf_path.stem}_json.json"
        print(f"Saving JSON output to: {output_json_path}")
        with open(output_json_path, "w") as json_file:
            json.dump(json_response, json_file, indent=2)

        try:
            # Annotate the PDF
            output_pdf_path = self.output_folder / f"{pdf_path.stem}_Annotated.pdf"
            print(f"Annotating PDF and saving to: {output_pdf_path}")
            draw_bounding_boxes(str(pdf_path), json_response, str(output_pdf_path))
            print(f"Finished processing {pdf_path.name}")
        except Exception as e:
            print(f"Error annotating PDF: {e}")

    def test_pdla_extraction(self, fast=False):
        url = f"{self.BASE_URL}/"
        print(f"Testing PDLA extraction at URL: {url}")
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(self.process_pdf, pdf_path, fast) for pdf_path in self.test_pdfs]
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions that occurred during execution

    def throughput_test(self, growth_func: GrowthFunc, start_page: int, end_page: int, num_pdfs: int, fast=False):
        print("Starting throughput test...")
        run_id = f"run_{uuid.uuid4().hex}"
        run_dir = self.input_folder / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created run directory: {run_dir}")

        original_pdf = self.test_pdfs[0]
        pdf_test_paths = []

        # Create a new PDF with only the specified page range
        base_pdf_writer = PdfWriter()
        with open(original_pdf, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num in range(start_page - 1, end_page):
                base_pdf_writer.add_page(pdf_reader.pages[page_num])

        base_pdf_path = run_dir / f"{original_pdf.stem}_base.pdf"
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

            test_pdf_name = f"{original_pdf.stem}_{multiplier}x.pdf"
            test_pdf_path = run_dir / test_pdf_name

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
        csv_path = self.output_folder / f"throughput_results_{growth_func.value}_{run_id}.csv"
        fieldnames = ['PDF Name', 'Number of Pages', 'Processing Time (seconds)', 'Throughput (pages/sec)']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            csvfile.flush()

            for pdf_path in pdf_test_paths:
                start_time = time.time()
                self.process_pdf(pdf_path)
                end_time = time.time()
                
                processing_time = end_time - start_time
                num_pages = len(PdfReader(pdf_path).pages)
                throughput = num_pages / processing_time
                
                row_data = {
                    'PDF Name': pdf_path.name,
                    'Number of Pages': num_pages,
                    'Processing Time (seconds)': processing_time,
                    'Throughput (pages/sec)': throughput
                }
                
                writer.writerow(row_data)
                csvfile.flush()  # Ensure data is written to file
                print(f"Processed {pdf_path.name}: {num_pages} pages in {processing_time:.2f} seconds. Throughput: {throughput:.2f} pages/sec")

        print(f"Throughput test results saved to {csv_path}")
        print("Throughput test completed successfully.")

    def test_parallel_requests(self, num_parallel_requests=5, fast=False):
        print("Starting parallel requests test...")
        input_folder = Path("input")
        pdf_files = list(input_folder.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError("No PDF files found in the input folder.")
        
        # Duplicate the first PDF file
        original_pdf = pdf_files[0]
        test_pdfs = [original_pdf] * num_parallel_requests
        
        def process_pdf(pdf_path):
            start_time = time.time()
            self.process_pdf(pdf_path, fast)
            end_time = time.time()
            processing_time = end_time - start_time
            return pdf_path.name, processing_time
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_requests) as executor:
            results = list(executor.map(process_pdf, test_pdfs))
        
        # Log results
        csv_path = self.output_folder / f"parallel_requests_results_{num_parallel_requests}_requests.csv"
        fieldnames = ['PDF Name', 'Processing Time (seconds)']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pdf_name, processing_time in results:
                row_data = {
                    'PDF Name': pdf_name,
                    'Processing Time (seconds)': processing_time
                }
                writer.writerow(row_data)
                print(f"Processed {pdf_name} in {processing_time:.2f} seconds.")
        
        print(f"Parallel requests test results saved to {csv_path}")
        print("Parallel requests test completed successfully.")

if __name__ == "__main__":
    tester = TestPDLAServer()
    tester.setUp()
    # Example of running throughput_test with user-provided parameters
    tester.test_parallel_requests(num_parallel_requests=10, fast=True)