import json
import os
import base64
from pathlib import Path
import argparse
from PIL import Image
import io
import shutil
from tqdm import tqdm
import concurrent.futures # Import futures

# Helper function to process a single line in parallel
def process_line_worker(args):
    line, line_idx, image_dir, table_id_field = args
    try:
        entry = json.loads(line.strip())
        conversations = []
        image_files = []
        table_id = entry.get(table_id_field, f"sample_{line_idx}")

        for message_idx, message in enumerate(entry.get('messages', [])):
            role = message.get('role', '')
            content = message.get('content', [])

            if role == 'user':
                prompt_parts = []
                image_saved_for_user = False # Track if image was saved for this user message
                for item in content:
                    if item.get('type') == 'text':
                        prompt_parts.append(item.get('text', ''))
                    elif item.get('type') == 'image_base64':
                        img_data = item.get('image_base64', '')
                        img_filename = f"{table_id}.jpg" # Assume one image per sample ID
                        img_path = os.path.join(image_dir, img_filename)

                        try:
                            # Avoid re-saving if already done for this table_id by another worker (less likely with unique IDs)
                            # A more robust check might involve checking file existence, but can slow down.
                            # For simplicity, we assume table_id is unique enough.
                            img_binary = base64.b64decode(img_data)
                            img = Image.open(io.BytesIO(img_binary))
                            img.save(img_path) # Overwrite if exists, should be same image
                            image_files.append(img_filename)
                            prompt_parts.append("<image>")
                            image_saved_for_user = True
                        except Exception as e:
                            # print(f"Error saving image {table_id} in worker: {e}") # Debug print
                            return None, "image_error" # Signal image save error

                if not image_saved_for_user and "<image>" not in prompt_parts:
                     # If user message was supposed to have an image but failed or was missing
                     pass # Or potentially return None, "missing_image_data"

                conversations.append({
                    "from": "human",
                    "value": "\n".join(prompt_parts)
                })

            elif role == 'assistant':
                response = ""
                for item in content:
                    if item.get('type') == 'text':
                        response += item.get('text', '')
                conversations.append({
                    "from": "gpt",
                    "value": response
                })

        # Validate result before returning
        if not image_files or len(conversations) < 2:
            return None, "skipped_no_image_or_conv"

        llava_entry = {
            "id": table_id,
            "conversations": conversations
        }
        if len(image_files) == 1:
            llava_entry["image"] = image_files[0]
        elif len(image_files) > 1:
             # This case might need review depending on how multiple images per entry are handled downstream
            llava_entry["image"] = image_files

        return llava_entry, "success"

    except json.JSONDecodeError:
        return None, "json_error"
    except Exception as e:
        # print(f"Generic error processing line {line_idx}: {e}") # Debug print
        return None, "processing_error"


def convert_to_llava_format(input_jsonl, output_json, image_dir, table_id_field="table_id", num_workers=8):
    """
    Convert the dataset from the current format to LLaVA format using parallel processing.
    """
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    llava_data = []
    metrics = {"total": 0, "success": 0, "skipped": 0, "errors": 0}

    with open(input_jsonl, 'r') as f, \
         concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        lines = list(f)
        metrics["total"] = len(lines)

        # Prepare tasks
        tasks = [(line, idx, image_dir, table_id_field) for idx, line in enumerate(lines)]

        # Process lines in parallel
        results = list(tqdm(executor.map(process_line_worker, tasks),
                            total=len(tasks),
                            desc=f"Converting {os.path.basename(input_jsonl)}"))

        # Collect results
        for entry, status in results:
            if status == "success" and entry:
                llava_data.append(entry)
                metrics["success"] += 1
            elif status in ["skipped_no_image_or_conv", "missing_image_data"]:
                 metrics["skipped"] += 1
            else: # json_error, image_error, processing_error
                 metrics["errors"] += 1


    # Write output file sequentially after collecting all results
    with open(output_json, 'w') as f:
        json.dump(llava_data, f, indent=2)

    print(f"Conversion complete: {metrics['success']}/{metrics['total']} successful, {metrics['skipped']} skipped, {metrics['errors']} errors")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to LLaVA format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory to save extracted images")
    parser.add_argument("--table_id_field", type=str, default="table_id", help="Field name for table ID in input data")
    parser.add_argument("--create_tensorboard_dir", action="store_true", help="Create tensorboard logs directory")
    parser.add_argument("--parallel", type=int, default=16, help="Number of parallel workers for conversion (increased default)") # Added parallel arg
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.image_dir, exist_ok=True)
    
    if args.create_tensorboard_dir:
        tensorboard_dir = os.path.join(args.output_dir, "tensorboard_logs")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Created tensorboard logs directory: {tensorboard_dir}")
    
    # Track overall metrics
    all_metrics = {}
    
    for split in ["train", "val", "test"]:
        input_file = os.path.join(args.input_dir, f"{split}.jsonl")
        output_file = os.path.join(args.output_dir, f"{split}.json")
        
        if os.path.exists(input_file):
            # print(f"Converting {split} set...")
            metrics = convert_to_llava_format(
                input_file, 
                output_file, 
                args.image_dir,
                args.table_id_field,
                num_workers=args.parallel # Pass parallel arg
            )
            all_metrics[split] = metrics
        else:
            print(f"Warning: {input_file} not found. Skipping.")
    
    # Write conversion summary
    summary_file = os.path.join(args.output_dir, "conversion_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\nConversion Summary:")
    for split, metrics in all_metrics.items():
        print(f"  {split}: {metrics['success']}/{metrics['total']} samples converted ({metrics['skipped']} skipped)")
    
    # Ensure the dataset_path.txt exists for other scripts
    with open(os.path.join(args.output_dir, "dataset_path.txt"), "w") as f:
        f.write(args.output_dir)

    print(f"\nOutput directory: {args.output_dir}")
    print(f"Image directory: {args.image_dir}")
    print(f"Summary file: {summary_file}")
    print("\nTo run evaluation after training:")
    print(f"  python src/training/evaluation.py --model_path ./output/my_lora_model --data_path {args.output_dir}/test.json --image_folder {args.image_dir}")

if __name__ == "__main__":
    main() 