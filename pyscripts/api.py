import dotenv
import os
import requests
from models import Model, TableOcr, TaskResponse, UploadForm, OcrStrategy
import time
import glob

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

def extract_file(file_to_send, model: Model, target_chunk_length: int = 512, ocr_strategy: OcrStrategy = OcrStrategy.Auto) -> TaskResponse:
    url = get_base_url() + "/api/v1/task"
    with open(file_to_send, "rb") as file:

        headers = get_headers()
        files = {"file": (os.path.basename(file_to_send), file, "application/pdf")}
        data = {
            "model": str(model.value),
            "target_chunk_length": str(target_chunk_length),
            "ocr_strategy": str(ocr_strategy.value)
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
    print(response.json())
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

def process_file(upload_form: UploadForm) -> TaskResponse:
    health_check()
    file_path = upload_form.file
    model = upload_form.model
    target_chunk_length = upload_form.target_chunk_length
    ocr_strategy = upload_form.ocr_strategy
    task = extract_file(file_path, model, target_chunk_length, ocr_strategy)
    print(f"Task id: {task.task_id}")
    task = check_task_status(task.task_url)
    # print(f"Task completed for {file_path}:")
    # print(task)
    return task

def process_all_files_in_input_folder(model: Model, table_ocr: TableOcr = None, target_chunk_length: int = 512, ocr_strategy: OcrStrategy = OcrStrategy.Auto):
    input_folder = "input"
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
    
    for file_path in pdf_files:
        print(f"Processing file: {file_path}")
        try:
            task = process_file(file_path, model, table_ocr, target_chunk_length, ocr_strategy)
            print(f"Task completed for {file_path}:")
            print(task)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    model = Model.HighQuality
    process_all_files_in_input_folder(model)