import math
from os import makedirs
from os.path import join, exists
from huggingface_hub import snapshot_download

MODELS_PATH="models"

def monitor_download_progress(downloaded_chunks, chunk_size, file_size):
    total_chunks = file_size // chunk_size
    progress_step = total_chunks // 5
    percent_complete = downloaded_chunks * chunk_size * 100 / file_size
    if downloaded_chunks % progress_step == 0:
        print(f"Downloaded {math.ceil(percent_complete)}%")

def download_required_models(model_id: str):
    makedirs(MODELS_PATH, exist_ok=True) 
    acquire_text_model()


def acquire_text_model():
    target_path = join(MODELS_PATH, "layoutlm-base-uncased")
    if exists(target_path):
        return
    makedirs(target_path, exist_ok=True)
    print("Embedding model is being downloaded")
    snapshot_download(repo_id="microsoft/layoutlm-base-uncased", local_dir=target_path, local_dir_use_symlinks=False)


if __name__ == "__main__":
    download_required_models("doclaynet")
