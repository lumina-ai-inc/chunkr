import dotenv
import os
import requests
from models import Model, TableOcr, TaskResponse, UploadForm, OcrStrategy
import time
import glob
import json  # Added import for JSON serialization

dotenv.load_dotenv(override=True)

def get_base_url():
    return os.getenv("INGEST_SERVER__URL")

def get_api_key():
    return os.getenv("INGEST_SERVER__API_KEY")

def get_headers():
    api_key = get_api_key()
    headers = {"Authorization": api_key}
    return headers

def health_check():
    url = get_base_url() + "/health"
    response = requests.get(url, headers=get_headers()).raise_for_status()
    return response

def extract_file(file_to_send, model: Model, target_chunk_length: int = 512, ocr_strategy: OcrStrategy = OcrStrategy.Auto, json_schema = None) -> TaskResponse:
    url = get_base_url() + "/api/v1/task"
    with open(file_to_send, "rb") as file:
        headers = get_headers()
        files = {
            "file": (os.path.basename(file_to_send), file, "application/pdf"),
        }
        
        if json_schema:
            # Serialize the json_schema to a JSON string
            json_schema_str = json.dumps(json_schema)
            files["json_schema"] = (None, json_schema_str, "application/json")
        
        data = {
            "model": str(model.value),
            "target_chunk_length": str(target_chunk_length),
            "ocr_strategy": str(ocr_strategy.value),
        }
        
        response = requests.post(url, files=files, data=data, headers=headers)
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
        try:
            task = get_task(url)
        except Exception as e:
            print(f"Error getting task status: {str(e)}")
    if task.status == "Failed":
        raise Exception(task.message)
    return task

def process_file(upload_form: UploadForm) -> TaskResponse:
    health_check()
    file_path = upload_form.file
    model = upload_form.model
    target_chunk_length = upload_form.target_chunk_length
    ocr_strategy = upload_form.ocr_strategy
    json_schema = upload_form.json_schema
    task = extract_file(file_path, model, target_chunk_length, ocr_strategy, json_schema)
    task = check_task_status(task.task_url)
    return task
