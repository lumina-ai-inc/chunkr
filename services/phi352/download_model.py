import os

from huggingface_hub import snapshot_download

# access_token = "<YOUR-HUGGINGFACE_TOKEN>"


if __name__ == "__main__":
    # download the meta/llama3 model
    os.makedirs("./models", exist_ok=True)
    snapshot_download(
        repo_id="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        local_dir="models",
        ignore_patterns=["*.pt", "*.bin"],
        #    token=access_token,
    )
