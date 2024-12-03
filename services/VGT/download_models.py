import math
from os import makedirs
from os.path import join, exists
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download, hf_hub_download

from configuration import service_logger, MODELS_PATH


def download_progress(count, block_size, total_size):
    total_counts = total_size // block_size
    show_counts_percentages = total_counts // 5
    percent = count * block_size * 100 / total_size
    if count % show_counts_percentages == 0:
        service_logger.info(f"Downloaded {math.ceil(percent)}%")


def download_vgt_model(model_name: str):
    service_logger.info(f"Downloading {model_name} model")
    model_path = join(MODELS_PATH, f"{model_name}_VGT_model.pth")
    if exists(model_path):
        return
    download_link = f"https://github.com/AlibabaResearch/AdvancedLiterateMachinery/releases/download/v1.3.0-VGT-release/{model_name}_VGT_model.pth"
    urlretrieve(download_link, model_path, reporthook=download_progress)


def download_models(model_name: str):
    makedirs(MODELS_PATH, exist_ok=True)

    download_vgt_model(model_name)
    download_embedding_model()

def download_embedding_model():
    model_path = join(MODELS_PATH, "layoutlm-base-uncased")
    if exists(model_path):
        return
    makedirs(model_path, exist_ok=True)
    service_logger.info("Embedding model is being downloaded")
    snapshot_download(repo_id="microsoft/layoutlm-base-uncased", local_dir=model_path, local_dir_use_symlinks=False)


if __name__ == "__main__":
    download_models("doclaynet")
