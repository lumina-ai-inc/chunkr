# finalize_and_upload.py
import argparse
import os
# Removed torch, transformers, peft imports as they are no longer needed for merging
from huggingface_hub import HfApi, upload_folder, HfFolder
# Removed Path import

# --- Removed Helper Function find_latest_checkpoint_or_adapter ---
# ... existing code ...

# --- Removed Custom merge function selective_merge_and_unload ---
# ... existing code ...

# --- Main Upload Logic ---
def main(args):
    print(f"--- Starting upload process for pre-merged model ---")

    merged_model_path = args.merged_model_dir
    if not os.path.isdir(merged_model_path):
        print(f"Error: Provided merged model directory does not exist: {merged_model_path}")
        return
    if not os.path.exists(os.path.join(merged_model_path, "config.json")):
        print(f"Error: 'config.json' not found in {merged_model_path}. Is it a valid model directory?")
        return
    # Add checks for other essential files if needed (e.g., model weights, processor files)
    print(f"Found pre-merged model directory: {merged_model_path}")


    # --- Removed base model loading ---
    # ... existing code ...

    # --- Removed LoRA adapter loading ---
    # ... existing code ...

    # --- Removed merging logic ---
    # ... existing code ...

    # --- Removed local saving logic ---
    # ... existing code ...


    print(f"Uploading pre-merged model from {merged_model_path} to Hub repository: {args.hub_repo_id}...")
    try:
        token = args.hf_token or HfFolder.get_token()
        if not token:
            raise ValueError("Hugging Face token not found. Please login using `huggingface-cli login` or provide --hf_token.")

        print(f"Using token: {'*' * (len(token) - 4)}{token[-4:]}") # Redacted for security

        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=args.hub_repo_id,
            repo_type="model",
            private=args.private_repo, # Use the new argument
            exist_ok=True
        )
        print(f"Ensured repository exists: {repo_url}")

        api.upload_folder(
            folder_path=merged_model_path, # Upload directly from the input directory
            repo_id=args.hub_repo_id,
            repo_type="model",
            commit_message=f"Upload pre-merged model from {merged_model_path}"
        )
        print(f"Successfully uploaded to: {repo_url}")
        print("--- Process Complete ---")

    except Exception as e:
        print(f"Error uploading pre-merged model to Hub: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a pre-merged model directory to Hugging Face Hub.")
    # Changed argument name and help text
    parser.add_argument("--merged_model_dir", type=str, required=True, help="Path to the directory containing the pre-merged model files.")
    # Removed base_model_id and output_dir arguments
    # parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model ID on Hugging Face Hub.")
    # parser.add_argument("--output_dir", type=str, required=True, help="Local path to save the merged model.")
    parser.add_argument("--hub_repo_id", type=str, default="ChunkrAI/sophris-table-qwen2_5VL-3B", help="Target Hugging Face Hub repo ID.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token.")
    # Removed safe_serialization argument
    # parser.add_argument("--safe_serialization", action='store_true', help="Use safetensors to save model.")
    parser.add_argument("--private_repo", action='store_true', help="Create the Hub repository as private.") # Added privacy argument

    args = parser.parse_args()
    main(args)
