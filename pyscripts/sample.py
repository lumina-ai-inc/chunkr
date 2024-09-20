import os, requests, time, glob, json
from dotenv import load_dotenv

load_dotenv(override=True)

def get_base_url(): return os.getenv("INGEST_SERVER__URL")
def get_headers(): return {"Authorization": os.getenv("INGEST_SERVER__API_KEY")}

def create_task(file_path):
    with open(file_path, "rb") as file:
        response = requests.post(f"{get_base_url()}/api/task",
            files={"file": (os.path.basename(file_path), file, "application/pdf")},
            data={"model": "HighQuality", "target_chunk_length": 0},
            headers=get_headers()
        )
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    return response.json()["task_url"]

def get_task(task_url): return requests.get(task_url, headers=get_headers()).json()

def save_to_json(output, file_name):
    output_json_path = os.path.join(os.path.dirname(__file__), "output", f"{file_name}_json.json")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    json.dump(output, open(output_json_path, "w"))
    return output_json_path

def process_files():
    for file_path in glob.glob(os.path.join("input", "*.pdf")):
        print(f"Processing file: {file_path}")
        try:
            task_url = create_task(file_path)
            while True:
                task = get_task(task_url)
                print(f"Task status: {task['status']}")
                if task["status"] == "Succeeded":
                    output = task.get("output")
                    if output is None: raise Exception(f"Output not found for {file_path}")
                    print(f"Downloading bounding boxes for {file_path}...")
                    output_json_path=save_to_json(output, os.path.basename(file_path).split(".")[0])
                    print(f"Downloaded json response to {output_json_path}")
                    break
                if task["status"] in ["Failed", "Canceled"]:
                    print(f"Task ended with status: {task['status']}")
                    break
                time.sleep(1)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__": process_files()