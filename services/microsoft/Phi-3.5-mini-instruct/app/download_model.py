
from huggingface_hub import snapshot_download
import os
# access_token = "<YOUR-HUGGINGFACE_TOKEN>"



if __name__ == "__main__":
    # download the meta/llama3 model
    os.makedirs("./models", exist_ok=True)
    snapshot_download(
        repo_id="microsoft/Phi-3.5-vision-instruct",
        local_dir="models",
        ignore_patterns=["*.pt", "*.bin"],
        #    token=access_token,
    )
