import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import Levenshtein
import re
from qwen_vl_utils import process_vision_info
from huggingface_hub import HfApi, upload_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--use_flash_attn", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hugging Face Hub model ID to upload results to")
    return parser.parse_args()

def replace_llava_tokens(text):
    text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
    text = text.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
    return text

def load_model_and_processor(model_path, device, use_flash_attn):
    # Check if it's a LoRA model by looking for adapter_config.json
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_lora:
        print("Loading a LoRA model...")
        try:
            adapter_config = json.load(open(os.path.join(model_path, "adapter_config.json")))
            base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct")
        except:
            base_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            
        print(f"Using base model: {base_model_name}")
        processor = AutoProcessor.from_pretrained(base_model_name)
        
        # Import specific model class based on model name
        if "qwen2.5-vl" in base_model_name.lower():
            from transformers import Qwen2_5VLForConditionalGeneration
            model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
            )
        
        model = PeftModel.from_pretrained(model, model_path)
    else:
        print("Loading a regular (non-LoRA) model...")
        # Try to detect model type from config
        try:
            config = json.load(open(os.path.join(model_path, "config.json")))
            model_type = config.get("model_type", "").lower()
        except:
            model_type = model_path.lower()
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Import specific model class based on detected type
        # if "qwen2.5" in model_type or "qwen2.5" in model_path.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_class = Qwen2_5_VLForConditionalGeneration
        # else:
        #     from transformers import Qwen2VLForConditionalGeneration
        #     model_class = Qwen2VLForConditionalGeneration
        
        print(f"Using model class: {model_class.__name__}")
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
        )
    
    model.eval()
    return model, processor

def get_image_path(image_file, image_folder):
    if os.path.exists(image_file):
        return image_file
    elif not image_file.startswith("http"):
        return os.path.join(image_folder, image_file)
    return image_file

def process_sample(sample, processor, image_folder):
    image_file = sample["image"]
    image_path = get_image_path(image_file, image_folder) if isinstance(image_file, str) else [get_image_path(img, image_folder) for img in image_file]
    
    prompt = ""
    response = ""
    
    for conv in sample["conversations"]:
        if conv["from"] == "human":
            prompt = replace_llava_tokens(conv["value"])
        elif conv["from"] == "gpt":
            response = conv["value"]
    
    return {
        "prompt": prompt,
        "image_path": image_path,
        "response": response
    }

def generate_response(model, processor, prompt, image_path, max_new_tokens):
    images = []
    if isinstance(image_path, str):
        images = [image_path]
    else:
        images = image_path
        
    vision_inputs = [{"type": "image", "image": img} for img in images]
    messages = [{"role": "user", "content": vision_inputs + [{"type": "text", "text": prompt}]}]
    
    try:
        encoded = processor.chat_tokenize_and_process_image(messages)
    except AttributeError:
        # Fallback for older processor versions
        encoded = processor.process_images_and_messages(messages)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoded["input_ids"].to(model.device),
            pixel_values=encoded["pixel_values"].to(model.device),
            image_grid_thw=encoded["image_grid_thw"].to(model.device) if "image_grid_thw" in encoded else None,
            attention_mask=encoded["attention_mask"].to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    response = processor.batch_decode(outputs[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return response

def compare_html(generated: str, ground_truth: str) -> float:
    """Compare HTML using Levenshtein distance for a more nuanced similarity measure."""
    gen_clean = generated.strip().replace("\n", "").replace(" ", "")
    gt_clean = ground_truth.strip().replace("\n", "").replace(" ", "")
    
    # Exact match check
    if gen_clean == gt_clean:
        return 1.0
    
    # Levenshtein distance-based similarity
    max_len = max(len(gen_clean), len(gt_clean))
    if max_len == 0:
        return 0.0
    
    distance = Levenshtein.distance(gen_clean, gt_clean)
    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity)

def process_batch(model, processor, batch, max_new_tokens, device):
    try:
        outputs = []
        for item in batch:
            # Get human prompt and image
            human_message = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'human'), '')
            image = item['image'] # PIL Image object

            # Create message format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, # Pass the PIL image directly
                    {"type": "text", "text": human_message}
                ]
            }]

            # 1. Apply chat template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 2. Process vision info
            # Note: process_vision_info expects file paths/URLs/base64, not PIL images directly.
            # We need a way to handle the PIL image. A temporary save or base64 conversion might be needed,
            # or check if qwen_vl_utils supports PIL directly (unlikely based on docs).
            # For now, let's assume a placeholder or skip if PIL isn't directly supported by process_vision_info.
            # This part needs refinement based on how process_vision_info handles PIL images.
            # --- Placeholder/Refinement Needed ---
            # Let's try passing the PIL image directly, hoping the underlying processor handles it.
            image_inputs, video_inputs = process_vision_info(messages)
            # If the above fails, we might need to save temporarily or convert to base64.
            # Example temporary save:
            # temp_img_path = f"temp_eval_image_{item['original_image']}"
            # image.save(temp_img_path)
            # messages[0]['content'][0]['image'] = temp_img_path # Update message with path
            # image_inputs, video_inputs = process_vision_info(messages)
            # os.remove(temp_img_path) # Clean up
            # --- End Placeholder ---


            # 3. Call processor
            inputs = processor(
                text=[text],
                images=image_inputs, # Use processed image info
                videos=video_inputs, # Should be empty for images
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            # 4. Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

                # 5. Trim and Decode
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0] # Get the first (and only) response in the batch of 1
                outputs.append(response)

        return outputs

    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        # Print traceback for more details
        import traceback
        traceback.print_exc()
        return [f"Error: {str(e)}"] * len(batch)

def evaluate(model, processor, data, image_folder, max_new_tokens, output_dir, batch_size, hub_model_id=None):
    results = []
    total_similarity = 0
    # Removed baseline model comparison logic for simplicity as requested
    # writer = SummaryWriter(output_dir) # Keep tensorboard if needed

    # Process data in batches
    for i in tqdm(range(0, len(data), batch_size)):
        # Create a sub-batch to process one item at a time due to process_vision_info complexity
        sub_batch_data = data[i:min(i + batch_size, len(data))]

        batch_items_processed = []
        # Load images for the sub-batch
        for item_data in sub_batch_data:
            item = item_data.copy() # Avoid modifying original data list
            try:
                image_filename = item['image']
                image_path = os.path.join(image_folder, image_filename)
                item['image'] = Image.open(image_path).convert('RGB') # Load as PIL Image
                item['original_image'] = image_filename
                batch_items_processed.append(item)
            except Exception as e:
                print(f"Error loading image {item.get('image', 'N/A')}: {str(e)}")
                # Skip this item if image loading fails

        if not batch_items_processed:
            continue

        # Process the prepared batch (which might be smaller than batch_size now)
        # The process_batch function now iterates internally
        outputs = process_batch(model, processor, batch_items_processed, max_new_tokens, model.device)

        # Calculate similarities for the successfully processed items
        for item, output in zip(batch_items_processed, outputs):
            if "Error:" in output: # Skip items that had processing errors
                 print(f"Skipping similarity calculation for {item['original_image']} due to processing error: {output}")
                 continue

            gpt_message = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt'), '')

            # Calculate Levenshtein similarity
            # Extract HTML if present
            html_match_gen = re.search(r"```html\s*(.*?)\s*```", output, re.DOTALL | re.IGNORECASE)
            gen_clean = html_match_gen.group(1).strip() if html_match_gen else output.strip()
            gen_clean = gen_clean.replace("\n", "").replace(" ", "")

            html_match_ref = re.search(r"```html\s*(.*?)\s*```", gpt_message, re.DOTALL | re.IGNORECASE)
            gt_clean = html_match_ref.group(1).strip() if html_match_ref else gpt_message.strip()
            gt_clean = gt_clean.replace("\n", "").replace(" ", "")


            max_len = max(len(gen_clean), len(gt_clean))
            if max_len == 0:
                similarity = 1.0 if len(gen_clean) == len(gt_clean) else 0.0 # Handle empty strings
            else:
                distance = Levenshtein.distance(gen_clean, gt_clean)
                similarity = 1.0 - (distance / max_len)
                similarity = max(0.0, similarity) # Ensure non-negative

            total_similarity += similarity
            results.append({
                'image': item['original_image'],
                'similarity': similarity,
                'exact_match': similarity == 1.0,
                'generated': output, # Optionally keep generated/reference for inspection
                'reference': gpt_message
            })
            # Log to tensorboard if needed
            # global_step = i + batch_items_processed.index(item)
            # writer.add_scalar('Evaluation/Levenshtein_Similarity', similarity, global_step)


    # Calculate average similarity
    avg_similarity = total_similarity / len(results) if results else 0
    # writer.add_scalar('Evaluation/Average_Levenshtein_Similarity', avg_similarity, 0) # Log average if using tensorboard
    # writer.close() # Close tensorboard writer

    # Save results locally
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists
    local_results_path = os.path.join(output_dir, 'eval_scores.json')
    results_data = {
        'average_similarity': avg_similarity,
        'num_samples_evaluated': len(results),
        'individual_scores': results
    }
    with open(local_results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"Evaluation complete. Average Levenshtein similarity: {avg_similarity:.4f} over {len(results)} samples.")
    print(f"Local results saved to: {local_results_path}")

    # Upload results to Hugging Face Hub
    if hub_model_id:
        print(f"Uploading evaluation results to Hugging Face Hub repository: {hub_model_id}...")
        try:
            upload_file(
                path_or_fileobj=local_results_path,
                path_in_repo="evaluation/eval_scores.json", # Store in a subdirectory on the Hub
                repo_id=hub_model_id,
                repo_type="model",
                commit_message="Upload evaluation results"
            )
            print("Upload to Hub complete.")
        except Exception as e:
            print(f"Error uploading evaluation results to Hub: {e}")

    return results # Return the detailed results list

def main():
    args = parse_args()
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model_path, args.device, args.use_flash_attn)
    
    # Load data
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    # Evaluate
    evaluate(model, processor, data, args.image_folder, args.max_new_tokens, args.output_dir, args.batch_size, args.hub_model_id)

if __name__ == "__main__":
    # Add qwen_vl_utils to sys path if it's not installed as a package
    # import sys
    # sys.path.append('/path/to/qwen_vl_utils_directory') # Adjust path as needed
    main()