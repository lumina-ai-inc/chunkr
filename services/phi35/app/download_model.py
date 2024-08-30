
from huggingface_hub import snapshot_download
import os
# access_token = "<YOUR-HUGGINGFACE_TOKEN>"


if __name__ == "__main__":
    # Download the Phi-3.5 Vision model
    os.makedirs("./models", exist_ok=True)
    snapshot_download(
        repo_id="microsoft/Phi-3.5-vision-instruct",
        local_dir="models",
        ignore_patterns=["*.pt", "*.bin"],
        #    token=access_token,
    )

    # Load the model without quantization
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "models",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2'
    )

    # Save the unquantized model
    model.save_pretrained("models/unquantized", safe_serialization=False)

    print("Unquantized model downloaded and saved to models/unquantized")