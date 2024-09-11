import os
from datetime import datetime
import concurrent.futures
from functools import partial
import glob

from api import process_file
from download import download_file
from models import Model, TableOcr
from annotate import draw_bounding_boxes


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


def extract_and_annotate_file(file_path: str, model: Model, table_ocr: TableOcr = None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(file_path).split(".")[0]
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    output_json_path = os.path.join(output_dir, f"{file_name}_json.json")
    output_annotated_path = os.path.join(output_dir, f"{file_name}_annotated.pdf")

    print(f"Processing file: {file_path}")
    task = process_file(file_path, model, table_ocr)
    output_url = task.output_file_url
    print(f"File processed: {file_path}")

    if output_url is None:
        raise Exception(f"File URL not found for {file_path}")

    print(f"Downloading bounding boxes for {file_path}...")
    json_path = download_file(output_url, output_json_path)
    print(f"Downloaded bounding boxes for {file_path}")

    print(f"Annotating file: {file_path}")
    draw_bounding_boxes(file_path, json_path, output_annotated_path)
    print(f"File annotated: {file_path}")


def process_all_files_in_input_folder(model: Model, table_ocr: TableOcr = None, max_workers=8):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, "input")
    
    # Get all PDF files in the input directory
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))

    # Create a partial function with the model parameter
    extract_func = partial(extract_and_annotate_file, model=model, table_ocr=table_ocr)

    def process_file_with_error_handling(file_path):
        try:
            extract_func(file_path)
            print(f"Successfully processed: {file_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")

    # Use ThreadPoolExecutor to parallelize the process
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_file_with_error_handling, file) for file in pdf_files]
        
        # As each task completes, print its result
        for future in concurrent.futures.as_completed(futures):
            # The result is None, but this will raise any exceptions that occurred
            future.result()


if __name__ == "__main__":
    model = Model.Fast
    table_ocr = None  # You can set this to None if you don't want to use table OCR
    process_all_files_in_input_folder(model, table_ocr)
    print("All files processed and annotated.")
