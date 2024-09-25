import os
import time
import csv
import uuid
import glob
from enum import Enum
import numpy as np
from PyPDF2 import PdfReader, PdfWriter

from main import main, GrowthFunc
from models import Model, TableOcr, OcrStrategy

def throughput_test(
    growth_func: GrowthFunc,
    start_page: int,
    end_page: int,
    num_pdfs: int,
    model: Model,
    target_chunk_length: int = None,
    ocr_strategy: OcrStrategy = OcrStrategy.Auto
):
    print("Starting throughput test...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input")
    run_id = f"run_{uuid.uuid4().hex}"
    run_dir = os.path.join(input_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")

    # Select only the first PDF from the input directory
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    if not pdf_files:
        raise ValueError("No PDF files found in the input folder.")
    original_pdf = pdf_files[0]  # Use only the first PDF
    pdf_test_paths = []

    # Create a new PDF with only the specified page range from the first PDF
    base_pdf_writer = PdfWriter()
    with open(original_pdf, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(start_page - 1, min(end_page, len(pdf_reader.pages))):
            base_pdf_writer.add_page(pdf_reader.pages[page_num])

    base_pdf_path = os.path.join(run_dir, f"{os.path.basename(original_pdf).split('.')[0]}_base.pdf")
    with open(base_pdf_path, 'wb') as output_file:
        base_pdf_writer.write(output_file)

    # Create all test PDFs by duplicating the base PDF according to the growth function
    for i in range(1, num_pdfs + 1):
        if growth_func == GrowthFunc.LINEAR:
            multiplier = i
        elif growth_func == GrowthFunc.EXPONENTIAL:
            multiplier = 2 ** (i - 1)
        elif growth_func == GrowthFunc.LOGARITHMIC:
            multiplier = max(1, int(np.log2(i + 1)))
        elif growth_func == GrowthFunc.QUADRATIC:
            multiplier = i ** 2
        elif growth_func == GrowthFunc.CUBIC:
            multiplier = i ** 3
        else:
            raise ValueError("Unsupported growth function")

        pdf_dir = os.path.join(run_dir, f"pdf_{multiplier}x")
        os.makedirs(pdf_dir, exist_ok=True)

        test_pdf_name = f"{os.path.basename(original_pdf).split('.')[0]}_{multiplier}x.pdf"
        test_pdf_path = os.path.join(pdf_dir, test_pdf_name)

        # Create the test PDF by duplicating the base PDF based on the multiplier
        test_pdf_writer = PdfWriter()
        for _ in range(multiplier):
            with open(base_pdf_path, 'rb') as base_file:
                base_reader = PdfReader(base_file)
                for page in base_reader.pages:
                    test_pdf_writer.add_page(page)

        with open(test_pdf_path, 'wb') as output_file:
            test_pdf_writer.write(output_file)

        print(f"Created {test_pdf_path} with multiplier {multiplier}x")
        pdf_test_paths.append(pdf_dir)

    # Process all created PDFs and log results
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"throughput_results_{growth_func.value}_{run_id}.csv")
    fieldnames = ['PDF Name', 'Number of Pages', 'Processing Time (seconds)', 'Throughput (pages/sec)']
    print(f"pdf_test_paths: {pdf_test_paths}")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        csvfile.flush()

        for pdf_dir in pdf_test_paths:
            start_time = time.time()
            try:
                # Pass the specific pdf_dir to the main function
                main(1, model, target_chunk_length, ocr_strategy, pdf_dir)
            except Exception as e:
                print(f"Failed to process {pdf_dir}: {str(e)}")
                continue
            end_time = time.time()

            processing_time = end_time - start_time
            pdf_path = glob.glob(os.path.join(pdf_dir, "*.pdf"))[0]  # Get the PDF file in the directory
            num_pages = len(PdfReader(pdf_path).pages)
            throughput = num_pages / processing_time if processing_time > 0 else float('inf')

            row_data = {
                'PDF Name': os.path.basename(pdf_path),
                'Number of Pages': num_pages,
                'Processing Time (seconds)': f"{processing_time:.2f}",
                'Throughput (pages/sec)': f"{throughput:.2f}"
            }

            writer.writerow(row_data)
            csvfile.flush()
            print(f"Processed {os.path.basename(pdf_path)}: {num_pages} pages in {processing_time:.2f} seconds. Throughput: {throughput:.2f} pages/sec")

    print(f"Throughput test results saved to {csv_path}")
    print("Throughput test completed successfully.")

if __name__ == "__main__":
    model = Model.HighQuality
    target_chunk_length = 1000  
    ocr_strategy = OcrStrategy.Off
    throughput_test(
        growth_func=GrowthFunc.LINEAR,
        start_page=1,
        end_page=10,
        num_pdfs=2,
        model=model,
        target_chunk_length=target_chunk_length,
        ocr_strategy=ocr_strategy
    )
    print("Throughput test completed.")