import dotenv
import os
import requests
from models import Model, TaskResponse
import time

dotenv.load_dotenv(override=True)


def get_base_url():
    return os.getenv("INGEST_SERVER__URL")


def get_api_key():
    return os.getenv("INGEST_SERVER__API_KEY")


def get_headers():
    api_key = get_api_key()
    headers = {"x-api-key": api_key}
    return headers


def health_check():
    url = get_base_url() + "/health"
    response = requests.get(url, headers=get_headers()).raise_for_status()
    return response


def extract_file(file_to_send, model: Model) -> TaskResponse:
    url = get_base_url() + "/api/task"
    with open(file_to_send, "rb") as file:
        file = {"file": (os.path.basename(file_to_send), file, "application/pdf")}
        file_data = {"model": model.value}
        headers = get_headers()
        response = requests.post(url, files=file, data=file_data, headers=headers)
        response.raise_for_status()
        return TaskResponse(**response.json())


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


if __name__ == "__main__":
    file_path = r"/Users/akhileshsharma/Documents/Lumina/backend/example-scripts/Extraction/input/apple_file_stamped_complaint_3.21.24.pdf"
    model = Model.Fast
    task = process_file(file_path, model)
    print(task)
