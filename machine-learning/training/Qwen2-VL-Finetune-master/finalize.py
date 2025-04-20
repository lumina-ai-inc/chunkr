# finalize_and_upload.py
import argparse
import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from huggingface_hub import HfApi, upload_folder, HfFolder
from pathlib import Path

# --- Helper Function (Optional but good practice) ---
def find_latest_checkpoint_or_adapter(model_path):
    path = Path(model_path)
    if (path / "adapter_config.json").exists():
        print(f"Found adapter config directly in: {model_path}")
        return str(path)

    checkpoints = sorted(
        [p for p in path.glob("checkpoint-*") if p.is_dir()],
        key=lambda x: int(x.name.split('-')[-1]) if x.name.split('-')[-1].isdigit() else -1,
        reverse=True
    )

    for ckpt_path in checkpoints:
        if (ckpt_path / "adapter_config.json").exists():
            print(f"Found latest valid adapter checkpoint: {ckpt_path}")
            return str(ckpt_path)

    raise FileNotFoundError(
        f"Could not find adapter_config.json directly in {model_path} or in any checkpoint-* subdirectory."
    )

# --- Custom merge for vLLM compatibility ---
def selective_merge_and_unload(peft_model, exclude_prefixes=("visual", "lm_head")):
    state_dict = peft_model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k.startswith(prefix) for prefix in exclude_prefixes)
    }

    base_state_dict = peft_model.base_model.model.state_dict()
    for k, v in filtered_state_dict.items():
        if 'lora_' in k:
            base_key = k.replace('base_model.model.', '').replace('.lora_A.weight', '').replace('.lora_B.weight', '')
            if base_key in base_state_dict:
                lora_A = state_dict[k.replace('lora_B.weight', 'lora_A.weight')]
                lora_B = state_dict[k.replace('lora_A.weight', 'lora_B.weight')]
                scaling = peft_model.peft_config[peft_model.active_adapter].lora_alpha / lora_A.shape[0]
                delta = (lora_B @ lora_A) * scaling
                base_state_dict[base_key] += delta

    peft_model.base_model.model.load_state_dict(base_state_dict, strict=False)
    return peft_model.base_model.model

# --- Main Merge and Upload Logic ---
def main(args):
    print(f"--- Starting merge and upload process ---")

    try:
        actual_adapter_path = find_latest_checkpoint_or_adapter(args.lora_adapter_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loading base model: {args.base_model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(
            args.base_model_id,
            token=args.hf_token,
            trust_remote_code=True
        )
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
            token=args.hf_token,
            trust_remote_code=True
        )
        print("Base model and processor loaded.")
    except Exception as e:
        print(f"Error loading base model/processor: {e}")
        return

    print(f"Loading LoRA adapter from: {actual_adapter_path}...")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            actual_adapter_path,
            device_map='cpu',
            token=args.hf_token
        )
        print("LoRA adapter loaded.")
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        return

    print("Merging LoRA weights (excluding vision/lm_head for vLLM)...")
    try:
        merged_model = selective_merge_and_unload(model)
        print("Selective merge complete.")
    except Exception as e:
        print(f"Error during selective merge: {e}")
        return

    print(f"Saving merged model and processor locally to: {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        merged_model.save_pretrained(args.output_dir, safe_serialization=args.safe_serialization)
        processor.save_pretrained(args.output_dir)
        print("Local save complete.")
    except Exception as e:
        print(f"Error saving model/processor locally: {e}")
        return

    print(f"Uploading merged model to Hub repository: {args.hub_repo_id}...")
    try:
        token = args.hf_token or HfFolder.get_token()
        if not token:
            raise ValueError("Hugging Face token not found. Please login using `huggingface-cli login` or provide --hf_token.")

        print(f"Using token: {'*' * (len(token) - 4)}{token[-4:]}")

        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=args.hub_repo_id,
            repo_type="model",
            private=True,
            exist_ok=True
        )
        print(f"Ensured repository exists: {repo_url}")

        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.hub_repo_id,
            repo_type="model",
            commit_message=f"Upload vLLM-compatible merged model from {args.lora_adapter_path}"
        )
        print(f"Successfully uploaded to: {repo_url}")
        print("--- Process Complete ---")

    except Exception as e:
        print(f"Error uploading merged model to Hub: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter with a base model and upload to Hugging Face Hub (vLLM compatible).")
    parser.add_argument("--lora_adapter_path", type=str, required=True, help="Path to the directory containing LoRA adapter checkpoints.")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model ID on Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str, required=True, help="Local path to save the merged model.")
    parser.add_argument("--hub_repo_id", type=str, default="ChunkrAI/sophris-table-qwen2_5VL-3B", help="Target Hugging Face Hub repo ID.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token.")
    parser.add_argument("--safe_serialization", action='store_true', help="Use safetensors to save model.")

    args = parser.parse_args()
    main(args)
