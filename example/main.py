import os
from datetime import datetime
import concurrent.futures
from functools import partial

from api import process_file
from download import download_file
from models import Model
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


def extract_and_annotate_file(file_path: str, model: Model):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(file_path).split(".")[0]
    output_dir = f"{current_dir}/output/{file_name}-{model.value}"
    output_json_path = f"{output_dir}/bounding_boxes.json"
    output_annotated_path = f"{output_dir}/annotated.pdf"

    print("Processing file...")
    task = process_file(file_path, model)
    task_url = task.file_url
    print("File processed")

    if task_url is None:
        raise Exception("File URL not found")

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading bounding boxes...")
    json_path = download_file(task_url, output_json_path)
    print("Downloaded bounding boxes")

    print("Annotating file...")
    draw_bounding_boxes(file_path, json_path, output_annotated_path)
    print("File annotated")


def extract_and_annotate_dir(dir_path: str, model: Model, max_workers=4):
    # Create a partial function with the model parameter
    extract_func = partial(extract_and_annotate_file, model=model)

    # Get full file paths
    file_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]

    def process_file_with_error_handling(file_path):
        try:
            extract_func(file_path)
            print(f"Successfully processed: {file_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")

    # Use ThreadPoolExecutor to parallelize the process
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and wait for them to complete
        list(executor.map(process_file_with_error_handling, file_paths))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, "input")
    model = Model.HighQuality
    extract_and_annotate_dir(dir_path, model)
