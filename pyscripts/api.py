import dotenv
import os
import requests
from models import Model, TaskResponse
import time
import glob

dotenv.load_dotenv(override=True)

def get_base_url():
    return os.getenv("INGEST_SERVER__URL")

def get_access_token():
    return os.getenv("INGEST_SERVER__ACCESS_TOKEN")

def get_headers():
    api_key = get_api_key()
    headers = {"Authorization": api_key}
    return headers

def health_check():
    url = get_base_url() + "/health"
    response = requests.get(url, headers=get_headers()).raise_for_status()
    return response

def extract_file(file_to_send, model: Model) -> TaskResponse:
    url = get_base_url() + "/api/task"
    with open(file_to_send, "rb") as file:
        file = {"file": (os.path.basename(file_to_send), file, "application/pdf")}
        file_data = {"model": model.value, "target_chunk_length": 100}
        headers = get_headers()
        response = requests.post(url, files=file, data=file_data, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        # Ensure configuration is included in the response
        if 'configuration' not in response_data:
            raise ValueError("Missing 'configuration' in response")
        return TaskResponse(**response_data)

def get_task(url: str) -> TaskResponse:
    headers = get_headers()
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    task = TaskResponse(**response.json())
    return task

def check_task_status(url: str) -> TaskResponse:
    task = get_task(url)
    while task.status == "Processing" or task.status == "Starting":
        time.sleep(1)
        task = get_task(url)
    if task.status == "Failed":
        raise Exception(task.message)
    return task

def process_file(file_path: str, model: Model) -> TaskResponse:
    health_check()
    task = extract_file(file_path, model)
    task = check_task_status(task.task_url)
    return task

def process_all_files_in_input_folder(model: Model):
    input_folder = "input"
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
    
    for file_path in pdf_files:
        print(f"Processing file: {file_path}")
        try:
            task = process_file(file_path, model)
            print(f"Task completed for {file_path}:")
            print(task)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    model = Model.Fast
    process_all_files_in_input_folder(model)