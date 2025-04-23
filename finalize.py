from src.merge_lora_weights import merge_lora
import argparse
import os

def main(args):
    print(f"--- Starting merge and upload process ---")

    # Create a compatible args object for merge_lora
    merge_args = argparse.Namespace(
        model_path=args.lora_adapter_path,
        model_base=args.base_model_id,
        save_model_path=args.output_dir,
        safe_serialization=args.safe_serialization,
        hub_model_id=args.hub_repo_id
    )

    # Use the tested merge_lora function
    merge_lora(merge_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and upload using merge_lora_weights.py")
    parser.add_argument("--lora_adapter_path", type=str, required=True, 
                       help="Path to the LoRA adapter checkpoint directory")
    parser.add_argument("--base_model_id", type=str, 
                       default="Qwen/Qwen2.5-VL-3B-Instruct",
                       help="Path or name of the base model")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to save the merged model")
    parser.add_argument("--safe_serialization", action='store_true',
                       help="Use safetensors for saving")
    parser.add_argument("--hub_repo_id", type=str,
                       default="ChunkrAI/sophris-table-qwen2_5VL-3B",
                       help="Hugging Face Hub model ID to upload to")
    
    args = parser.parse_args()
    main(args) 