import json
import os
import base64
from pathlib import Path
import argparse
from PIL import Image
import io
import shutil
from tqdm import tqdm

def convert_to_llava_format(input_jsonl, output_json, image_dir, table_id_field="table_id"):
    """
    Convert the dataset from the current format to LLaVA format.
    
    Args:
        input_jsonl: Path to input JSONL file
        output_json: Path to output JSON file
        image_dir: Directory to save extracted images
        table_id_field: Field name for the table ID in input data
    """
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    
    llava_data = []
    metrics = {"total": 0, "success": 0, "skipped": 0}
    
    with open(input_jsonl, 'r') as f:
        lines = list(f)
        for line_idx, line in enumerate(tqdm(lines, desc=f"Converting {os.path.basename(input_jsonl)}")):
            metrics["total"] += 1
            entry = json.loads(line.strip())
            conversations = []
            image_files = []
            
            # Get table_id for consistent image naming (if available)
            table_id = entry.get(table_id_field, f"sample_{line_idx}")
            
            for message_idx, message in enumerate(entry.get('messages', [])):
                role = message.get('role', '')
                content = message.get('content', [])
                
                if role == 'user':
                    # Process user message
                    prompt_parts = []
                    for item in content:
                        if item.get('type') == 'text':
                            prompt_parts.append(item.get('text', ''))
                        elif item.get('type') == 'image_base64':
                            # Save image to file with consistent naming
                            img_data = item.get('image_base64', '')
                            img_filename = f"{table_id}.jpg"
                            img_path = os.path.join(image_dir, img_filename)
                            
                            # Decode and save image
                            try:
                                img_binary = base64.b64decode(img_data)
                                img = Image.open(io.BytesIO(img_binary))
                                img.save(img_path)
                                image_files.append(img_filename)
                                prompt_parts.append("<image>")
                            except Exception as e:
                                print(f"Error saving image {table_id}: {e}")
                                metrics["skipped"] += 1
                                continue
                    
                    conversations.append({
                        "from": "human",
                        "value": "\n".join(prompt_parts)
                    })
                
                elif role == 'assistant':
                    # Process assistant message
                    response = ""
                    for item in content:
                        if item.get('type') == 'text':
                            response += item.get('text', '')
                    
                    conversations.append({
                        "from": "gpt",
                        "value": response
                    })
            
            # Skip if no images found or no complete conversations
            if not image_files or len(conversations) < 2:
                metrics["skipped"] += 1
                continue
                
            # Create LLaVA format entry
            llava_entry = {
                "id": table_id,
                "conversations": conversations
            }
            
            # Handle single or multiple images
            if len(image_files) == 1:
                llava_entry["image"] = image_files[0]
            elif len(image_files) > 1:
                llava_entry["image"] = image_files
            
            llava_data.append(llava_entry)
            metrics["success"] += 1
    
    # Write output file
    with open(output_json, 'w') as f:
        json.dump(llava_data, f, indent=2)
    
    print(f"Conversion complete: {metrics['success']}/{metrics['total']} successful, {metrics['skipped']} skipped")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to LLaVA format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory to save extracted images")
    parser.add_argument("--table_id_field", type=str, default="table_id", help="Field name for table ID in input data")
    parser.add_argument("--create_tensorboard_dir", action="store_true", help="Create tensorboard logs directory")
    
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
                args.table_id_field
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