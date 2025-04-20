import argparse
from utils import get_model_name_from_path
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from huggingface_hub import HfApi, upload_folder
import os
import torch

def merge_lora(args):
    print(f"Loading base model: {args.model_base}...")
    processor = AutoProcessor.from_pretrained(args.model_base)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_base,
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    print("Base model loaded.")

    print(f"Loading LoRA adapter from: {args.model_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        args.model_path,
        device_map='cpu'
    )
    print("LoRA adapter loaded.")

    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("Merging complete.")

    print(f"Saving merged model to {args.save_model_path}...")
    model.save_pretrained(args.save_model_path, safe_serialization=args.safe_serialization)
    processor.save_pretrained(args.save_model_path)
    print("Local save complete.")

    if args.hub_model_id:
        print(f"Uploading merged model to Hugging Face Hub repository: {args.hub_model_id}...")
        try:
            upload_folder(
                folder_path=args.save_model_path,
                repo_id=args.hub_model_id,
                repo_type="model",
                commit_message=f"Upload merged model from {args.model_path}",
            )
            print("Upload to Hub complete.")
        except Exception as e:
            print(f"Error uploading merged model to Hub: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the LoRA adapter checkpoint directory")
    parser.add_argument("--model-base", type=str, required=True, help="Path or name of the base model (e.g., Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--save-model-path", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--safe-serialization", action='store_true', help="Use safetensors for saving")
    parser.add_argument("--hub-model-id", type=str, default=None, help="Hugging Face Hub model ID to upload the merged model to")

    args = parser.parse_args()

    os.makedirs(args.save_model_path, exist_ok=True)

    merge_lora(args)