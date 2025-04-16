import json
import os
import base64
from pathlib import Path
import argparse
from PIL import Image
import io

def convert_to_llava_format(input_jsonl, output_json, image_dir):
    """
    Convert the dataset from the current format to LLaVA format.
    
    Args:
        input_jsonl: Path to input JSONL file
        output_json: Path to output JSON file
        image_dir: Directory to save extracted images
    """
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    
    llava_data = []
    
    with open(input_jsonl, 'r') as f:
        for line_idx, line in enumerate(f):
            entry = json.loads(line.strip())
            conversations = []
            image_files = []
            
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
                            # Save image to file
                            img_data = item.get('image_base64', '')
                            img_id = f"{line_idx}_{message_idx}_{len(image_files)}"
                            img_filename = f"{img_id}.jpg"
                            img_path = os.path.join(image_dir, img_filename)
                            
                            # Decode and save image
                            try:
                                img_binary = base64.b64decode(img_data)
                                img = Image.open(io.BytesIO(img_binary))
                                img.save(img_path)
                                image_files.append(img_filename)
                                prompt_parts.append("<image>")
                            except Exception as e:
                                print(f"Error saving image {img_id}: {e}")
                    
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
            
            # Create LLaVA format entry
            llava_entry = {
                "id": f"sample_{line_idx}",
                "conversations": conversations
            }
            
            # Handle single or multiple images
            if len(image_files) == 1:
                llava_entry["image"] = image_files[0]
            elif len(image_files) > 1:
                llava_entry["image"] = image_files
            
            llava_data.append(llava_entry)
    
    # Write output file
    with open(output_json, 'w') as f:
        json.dump(llava_data, f, indent=2)
    
    print(f"Converted {len(llava_data)} samples to {output_json}")
    print(f"Images saved to {image_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to LLaVA format")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory to save extracted images")
    
    args = parser.parse_args()
    
    for split in ["train", "val", "test"]:
        input_file = os.path.join(args.input_dir, f"{split}.jsonl")
        output_file = os.path.join(args.output_dir, f"{split}.json")
        
        if os.path.exists(input_file):
            print(f"Converting {split} set...")
            convert_to_llava_format(input_file, output_file, args.image_dir)
        else:
            print(f"Warning: {input_file} not found. Skipping.")

if __name__ == "__main__":
    main() 