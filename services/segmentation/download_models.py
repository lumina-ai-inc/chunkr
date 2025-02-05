import math
from os import makedirs
from os.path import join, exists
from huggingface_hub import snapshot_download
from urllib.request import urlretrieve
from configuration import MODELS_PATH

def monitor_download_progress(downloaded_chunks, chunk_size, file_size):
    total_chunks = file_size // chunk_size
    progress_step = total_chunks // 5
    percent_complete = downloaded_chunks * chunk_size * 100 / file_size
    if downloaded_chunks % progress_step == 0:
        print(f"Downloaded {math.ceil(percent_complete)}%")

def download_required_models(model_id: str):
    print("Downloading models now1...")
    makedirs(MODELS_PATH, exist_ok=True) 
    print("Downloading models now2...")
    acquire_text_model()
    print("LayoutLM model downloaded successfully!")
    download_vgt_model(model_id)
    print("VGT model downloaded successfully!")

def acquire_text_model():
    target_path = join(MODELS_PATH, "layoutlm-base-uncased")
    if exists(target_path):
        print("LayoutLM model already exists")
        return
    makedirs(target_path, exist_ok=True)
    print("Embedding model is being downloaded")
    snapshot_download(repo_id="microsoft/layoutlm-base-uncased", local_dir=target_path, local_dir_use_symlinks=False)

def download_vgt_model(model_name: str):
    model_path = join(MODELS_PATH, f"{model_name}_VGT_model.pth")
    if exists(model_path):
        return
    download_link = f"https://github.com/AlibabaResearch/AdvancedLiterateMachinery/releases/download/v1.3.0-VGT-release/{model_name}_VGT_model.pth"
    urlretrieve(download_link, model_path, reporthook=monitor_download_progress)
    
if __name__ == "__main__":
    print("Downloading models...")
    download_required_models("doclaynet")
    print("Models downloaded successfully!")