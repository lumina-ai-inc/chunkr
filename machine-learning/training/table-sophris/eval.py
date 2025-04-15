import os
import torch
from unsloth import FastVisionModel, is_bf16_supported
from transformers import TextStreamer
from dotenv import load_dotenv
import logging
from PIL import Image
import random
import re
from typing import Optional
import Levenshtein
# import bitsandbytes as bnb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uuid
import datetime
import seaborn as sns
import json
import io
import base64

# Assuming data_loader.py is in the same directory or accessible
from data_loader import TableDatasetLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    return max(0.0, similarity)  # Ensure non-negative

# Modified function to handle both baseline and fine-tuned models
def run_evaluation(
    model_name_or_path: str,
    num_eval_samples: int,
    base_model_name: Optional[str] = None,
    batch_size: int = 4,
    output_folder: str = "",
    data_dir: str = "data",
    dataset_name: str = None,
    eval_split: str = "test",
    compare_with_baseline: bool = True,
    baseline_model: str = "unsloth/Qwen2.5-VL-3B-Instruct"
):
    """Loads a model (base or fine-tuned) and runs evaluation."""
    
    # Check if model_name_or_path is a relative path that should be joined with output_dir
    if model_name_or_path and not os.path.isabs(model_name_or_path) and not model_name_or_path.startswith("unsloth/"):
        # If it's a relative path, try to find it in standard locations
        potential_model_paths = [
            model_name_or_path,  # As provided
            os.path.join("outputs", model_name_or_path),  # In outputs dir
            os.path.join("outputs", f"lora_{model_name_or_path}"),  # With lora_ prefix
            os.path.join("outputs", f"lora_{base_model_name.replace('/', '_')}") if base_model_name else None,  # Based on base model
        ]
        
        for path in potential_model_paths:
            if path and os.path.exists(path):
                logger.info(f"Found model at: {path}")
                model_name_or_path = path
                break
                
        logger.info(f"Using model path: {model_name_or_path}")
    
    # Get dataset name from environment if not provided
    if dataset_name is None:
        dataset_name = os.environ.get("DATASET_NAME", "default")
    
    is_finetuned = base_model_name is not None
    eval_type = "Fine-tuned" if is_finetuned else "Baseline"
    model_id = model_name_or_path if not is_finetuned else f"{base_model_name} + LoRA: {model_name_or_path}"
    
    # Create a unique run folder name based on model and timestamp
    model_name_safe = model_name_or_path.replace("/", "_").replace("\\", "_")
    run_folder = f"{output_folder}/{model_name_safe}" if output_folder else f"runs/{model_name_safe}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_folder, exist_ok=True)
    logger.info(f"Saving outputs to: {run_folder}")

    # --- Configuration ---
    load_in_4bit = False
    max_new_tokens = 512
    temperature = 0.1
    min_p = 0.1

    # Data loading config
    s3_bucket = os.environ.get("S3_BUCKET")
    data_limit = num_eval_samples * 2

    # --- Load Model and Tokenizer ---
    try:
        if is_finetuned:
            logger.info(f"Loading FINE-TUNED model: LoRA from '{model_name_or_path}' on base '{base_model_name}'")
            if not os.path.exists(model_name_or_path):
                logger.error(f"LoRA adapter path not found: {model_name_or_path}")
                logger.error("Skipping evaluation for this fine-tuned model.")
                return
                
            # Load the base model first
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=base_model_name,
                use_gradient_checkpointing="unsloth",
            )
            
            # Now load the LoRA weights
            logger.info(f"Loading LoRA weights from {model_name_or_path}")
            from peft import PeftModel, PeftConfig
            
            try:
                # First try the simpler approach - load directly as PEFT model
                model = PeftModel.from_pretrained(model, model_name_or_path)
                logger.info("Successfully loaded fine-tuned model using PEFT")
            except Exception as e1:
                logger.warning(f"First loading attempt failed: {e1}")
                try:
                    # Alternative approach - use Unsloth's get_peft_model with load_adapter
                    config = PeftConfig.from_pretrained(model_name_or_path)
                    model = FastVisionModel.get_peft_model(
                        model,
                        r=16,  # These values don't matter as we'll load the saved config
                        lora_alpha=16,
                        lora_dropout=0,
                        bias="none",
                        use_gradient_checkpointing=False
                    )
                    model.load_adapter(model_name_or_path)
                    logger.info("Successfully loaded fine-tuned model using load_adapter")
                except Exception as e2:
                    logger.error(f"Failed to load fine-tuned model with second attempt: {e2}")
                    logger.error("Skipping evaluation for this model.")
                    del model
                    del tokenizer
                    torch.cuda.empty_cache()
                    return
        else:
            logger.info(f"Loading BASE model: '{model_name_or_path}'")
            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_name_or_path,
                use_gradient_checkpointing="unsloth",
            )
            
        logger.info("Model loaded successfully.")
        FastVisionModel.for_inference(model)
    except Exception as e:
        logger.error(f"Failed to load model '{model_id}': {e}")
        logger.error("Skipping evaluation for this model.")
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        torch.cuda.empty_cache()
        return

    # --- Load Evaluation Data ---
    logger.info(f"Loading evaluation data from {eval_split} split...")
    
    # Look in possible locations for the split file
    split_file_paths = [
        os.path.join(data_dir, dataset_name, f"{eval_split}.jsonl"),  # New structure
        os.path.join(data_dir, f"{eval_split}.jsonl"),                # Old structure
        os.path.join(data_dir, dataset_name, "jsonls", f"{eval_split}.jsonl")  # Alternative structure
    ]
    
    split_file = None
    for path in split_file_paths:
        if os.path.exists(path):
            split_file = path
            logger.info(f"Found {eval_split} split at {split_file}")
            break
    
    if split_file is None:
        logger.error(f"Could not find {eval_split} split file in any of the expected locations:")
        for path in split_file_paths:
            logger.error(f"- {path}")
        logger.error("Please ensure dataset is prepared correctly with prepare_dataset.py")
        return None

    # Load the JSONL directly
    raw_samples = []
    with open(split_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extract base64 image and convert back to PIL
                for message in data.get('messages', []):
                    for content in message.get('content', []):
                        if content.get('type') == 'image_base64' and 'image_base64' in content:
                            img_data = base64.b64decode(content['image_base64'])
                            image = Image.open(io.BytesIO(img_data))
                            
                            # Get HTML content from assistant response
                            html_content = ""
                            for ass_message in data.get('messages', []):
                                if ass_message.get('role') == 'assistant':
                                    for ass_content in ass_message.get('content', []):
                                        if ass_content.get('type') == 'text':
                                            text = ass_content.get('text', '')
                                            # Extract HTML from markdown code block
                                            html_match = re.search(r"```html\s*(.*?)\s*```", text, re.DOTALL)
                                            if html_match:
                                                html_content = html_match.group(1).strip()
                            
                            # Create a sample object similar to what TableDatasetLoader would return
                            sample = type('Sample', (), {
                                'image': image,
                                'html': html_content,
                                'table_id': data.get('table_id', f"sample_{len(raw_samples)}")
                            })
                            raw_samples.append(sample)
                            break
            except Exception as e:
                logger.error(f"Error processing {eval_split} sample: {e}")
    
    logger.info(f"Loaded {len(raw_samples)} samples from {eval_split} split")

    if len(raw_samples) < num_eval_samples:
        logger.warning(f"Could only load {len(raw_samples)} samples for evaluation (requested {num_eval_samples}).")
        if not raw_samples:
            logger.error("No evaluation samples loaded. Exiting evaluation for this model.")
            del model
            del tokenizer
            torch.cuda.empty_cache()
            return
    raw_samples = raw_samples[:num_eval_samples]

    logger.info(f"Loaded {len(raw_samples)} samples for evaluation.")

    # --- Run Evaluation ---
    logger.info("Starting evaluation run...")
    total_similarity = 0.0
    instruction = "OCR the table and convert it to HTML. Output the HTML directly in ```html``` tags."
    
    results = []

    # Process samples in batches
    for batch_idx in range(0, len(raw_samples), batch_size):
        batch_samples = raw_samples[batch_idx:batch_idx + batch_size]
        logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(raw_samples) + batch_size - 1)//batch_size}")
        
        batch_inputs = []
        batch_images = []
        
        for sample in batch_samples:
            image = sample.image
            batch_images.append(image)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            try:
                inputs = tokenizer(
                    images=image, text=input_text, add_special_tokens=False, return_tensors="pt",
                ).to(model.device)
                batch_inputs.append(inputs)
            except Exception as e:
                logger.error(f"Error tokenizing input for sample {sample.table_id}: {e}")
                batch_inputs.append(None)
        
        # Process each sample in the batch
        for i, (sample, inputs) in enumerate(zip(batch_samples, batch_inputs)):
            if inputs is None:
                continue
                
            sample_idx = batch_idx + i
            logger.info(f"--- Evaluating Sample {sample_idx+1}/{len(raw_samples)} (ID: {sample.table_id}) ---")
            ground_truth_html = sample.html
            
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, use_cache=True,
                        temperature=temperature, min_p=min_p, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    )
            except Exception as e:
                logger.error(f"Error during model generation for sample {sample.table_id}: {e}")
                continue

            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            generated_text_raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Extract HTML content - try multiple patterns
            generated_html = ""
            # Try to find content within ```html``` tags
            match = re.search(r"```html\s*(.*?)\s*```", generated_text_raw, re.DOTALL | re.IGNORECASE)
            if match:
                generated_html = match.group(1).strip()
            else:
                # Try to find content within <table>...</table> tags
                match = re.search(r"(<table>.*?</table>)", generated_text_raw, re.DOTALL | re.IGNORECASE)
                if match:
                    generated_html = match.group(1).strip()
                else:
                    # If no HTML tags found, use the raw output
                    logger.warning(f"Could not find HTML content in generated output for {sample.table_id}. Using raw output.")
                    generated_html = generated_text_raw

            # Save the full raw output for debugging
            sample_folder = os.path.join(run_folder, f"sample_{sample_idx+1}_{sample.table_id}")
            os.makedirs(sample_folder, exist_ok=True)
            with open(os.path.join(sample_folder, "raw_output.txt"), "w") as f:
                f.write(generated_text_raw)

            similarity = compare_html(generated_html, ground_truth_html)
            total_similarity += similarity
            logger.info(f"HTML Similarity: {similarity:.4f}")
            
            # Save the image
            image_path = os.path.join(sample_folder, "input_image.png")
            batch_images[i].save(image_path)
            
            # Save generated HTML
            with open(os.path.join(sample_folder, "generated.html"), "w") as f:
                f.write(generated_html)
                
            # Save ground truth HTML
            with open(os.path.join(sample_folder, "ground_truth.html"), "w") as f:
                f.write(ground_truth_html)
                
            # Save comparison result
            with open(os.path.join(sample_folder, "comparison.txt"), "w") as f:
                f.write(f"Similarity score: {similarity:.4f}\n")
                f.write(f"Exact match: {'Yes' if similarity == 1.0 else 'No'}\n")
                if similarity < 1.0:
                    gen_clean = generated_html.strip().replace("\n", "").replace(" ", "")
                    gt_clean = ground_truth_html.strip().replace("\n", "").replace(" ", "")
                    f.write(f"Levenshtein distance: {Levenshtein.distance(gen_clean, gt_clean)}\n")
            
            results.append({
                "sample_id": sample.table_id,
                "similarity": similarity,
                "exact_match": similarity == 1.0
            })
            
        # Clear memory after each batch
        torch.cuda.empty_cache()

    if not raw_samples:
        logger.error("No samples were successfully evaluated for this model.")
        average_similarity = 0.0
    else:
        average_similarity = total_similarity / len(raw_samples)

    logger.info(f"\n--- {eval_type} Evaluation Summary (Model: {model_id}) ---")
    logger.info(f"Average HTML Similarity over {len(raw_samples)} samples: {average_similarity:.4f}")
    logger.info(f"{'='*20} Finished Evaluation for: {model_id} {'='*20}\n")

    # Save overall results
    with open(os.path.join(run_folder, "summary.txt"), "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Evaluation type: {eval_type}\n")
        f.write(f"Number of samples: {len(raw_samples)}\n")
        f.write(f"Average similarity: {average_similarity:.4f}\n\n")
        
        f.write("Per-sample results:\n")
        for result in results:
            f.write(f"Sample {result['sample_id']}: Similarity={result['similarity']:.4f}, Exact match={result['exact_match']}\n")
    
    # Create and save graphs
    if results:
        # Extract data for plotting
        sample_ids = [r['sample_id'] for r in results]
        similarities = [r['similarity'] for r in results]
        exact_matches = [1 if r['exact_match'] else 0 for r in results]
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Sample ID': sample_ids,
            'Similarity': similarities,
            'Exact Match': exact_matches
        })
        
        # Plot similarity scores
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(similarities)), similarities, color='skyblue')
        plt.axhline(y=average_similarity, color='red', linestyle='--', label=f'Average: {average_similarity:.4f}')
        plt.xlabel('Sample Index')
        plt.ylabel('Similarity Score')
        plt.title(f'HTML Similarity Scores - {model_id}')
        plt.ylim(0, 1.05)
        plt.xticks(range(len(similarities)), [f"{i+1}" for i in range(len(similarities))], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, "similarity_scores.png"))
        plt.close()
        
        # Plot exact match distribution
        exact_match_count = sum(exact_matches)
        plt.figure(figsize=(8, 8))
        plt.pie([exact_match_count, len(exact_matches) - exact_match_count], 
                labels=['Exact Match', 'Partial Match'], 
                autopct='%1.1f%%', 
                colors=['#66b3ff', '#ff9999'])
        plt.title(f'Exact vs Partial Matches - {model_id}')
        plt.savefig(os.path.join(run_folder, "exact_match_distribution.png"))
        plt.close()
        
        # Histogram of similarity scores
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=10, alpha=0.7, color='green')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Similarity Scores - {model_id}')
        plt.axvline(x=average_similarity, color='red', linestyle='--', label=f'Average: {average_similarity:.4f}')
        plt.legend()
        plt.savefig(os.path.join(run_folder, "similarity_histogram.png"))
        plt.close()
        
        # Save the data as CSV for further analysis
        df.to_csv(os.path.join(run_folder, "results_data.csv"), index=False)
    
    logger.info(f"Results and graphs saved to: {run_folder}")
    
    # Return results for potential aggregation
    result_dict = {
        "model_id": model_id,
        "eval_type": eval_type,
        "average_similarity": average_similarity,
        "num_samples": len(raw_samples),
        "results": results,
        "run_folder": run_folder
    }
    
    # Run baseline evaluation if needed
    baseline_result = None
    if is_finetuned and compare_with_baseline and baseline_model:
        logger.info(f"Running baseline evaluation on {baseline_model} for comparison...")
        baseline_folder = os.path.join(os.path.dirname(run_folder), f"baseline_{baseline_model.replace('/', '_')}")
        baseline_result = run_evaluation(
            model_name_or_path=baseline_model,
            num_eval_samples=num_eval_samples,
            base_model_name=None,
            batch_size=batch_size,
            output_folder=baseline_folder,
            data_dir=data_dir,
            dataset_name=dataset_name,
            eval_split=eval_split,
            compare_with_baseline=False  # Prevent infinite recursion
        )
        logger.info(f"Baseline evaluation complete. Average similarity: {baseline_result['average_similarity']:.4f}")

    # Add baseline comparison if available
    if baseline_result:
        result_dict["baseline_model_id"] = baseline_result["model_id"]
        result_dict["baseline_similarity"] = baseline_result["average_similarity"]
        result_dict["improvement"] = average_similarity - baseline_result["average_similarity"]
        
        # Add comparison to summary
        with open(os.path.join(run_folder, "baseline_comparison.txt"), "w") as f:
            f.write(f"Fine-tuned model: {model_id}\n")
            f.write(f"Fine-tuned similarity: {average_similarity:.4f}\n\n")
            f.write(f"Baseline model: {baseline_result['model_id']}\n")
            f.write(f"Baseline similarity: {baseline_result['average_similarity']:.4f}\n\n")
            f.write(f"Improvement: {result_dict['improvement']:.4f} ({result_dict['improvement']*100:.1f}%)\n")
    
    return result_dict

if __name__ == "__main__":
    load_dotenv(override=True)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate table OCR models")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate per model")
    parser.add_argument("--baseline_models", nargs="+", default=["unsloth/Qwen2.5-VL-3B-Instruct", "unsloth/Qwen2.5-VL-7B-Instruct"], 
                        help="List of baseline models to evaluate")
    parser.add_argument("--finetuned_path", type=str, default="lora_model/qwen2_5_Vl_3B-table-finetune", 
                        help="Path to finetuned LoRA model")
    parser.add_argument("--finetuned_base", type=str, default="unsloth/Qwen2.5-VL-3B-Instruct", 
                        help="Base model for the finetuned LoRA")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline model evaluation")
    parser.add_argument("--skip_finetuned", action="store_true", help="Skip finetuned model evaluation")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory containing train/val/test split data")
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test"],
                        help="Which data split to use for evaluation (val or test)")
    
    args = parser.parse_args()
    
    # Create a unique run ID
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"runs/eval_{timestamp}_{run_id}"
    os.makedirs(run_folder, exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(run_folder, "eval.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting evaluation run {run_id}")
    logger.info(f"Saving all outputs to: {run_folder}")
    
    all_results = []
    all_sample_results = []  # To track results across all samples and models

    if not args.skip_baseline:
        logger.info("--- Starting Baseline Model Evaluations ---")
        for model_name in args.baseline_models:
            model_folder = os.path.join(run_folder, model_name.replace("/", "_"))
            os.makedirs(model_folder, exist_ok=True)
            
            result = run_evaluation(
                model_name_or_path=model_name,
                num_eval_samples=args.num_samples,
                base_model_name=None,
                batch_size=args.batch_size,
                output_folder=model_folder,
                data_dir=args.data_dir,
                dataset_name=None,
                eval_split=args.eval_split,
                compare_with_baseline=False  # Skip comparing with baseline
            )
            if result:
                all_results.append(result)
                # Add model identifier to each sample result
                for sample_result in result["results"]:
                    all_sample_results.append({
                        "model_id": result["model_id"],
                        "model_type": result["eval_type"],
                        "sample_id": sample_result["sample_id"],
                        "similarity": sample_result["similarity"],
                        "exact_match": sample_result["exact_match"]
                    })

    if not args.skip_finetuned:
        logger.info("--- Starting Fine-tuned Model Evaluation ---")
        model_folder = os.path.join(run_folder, "finetuned_" + args.finetuned_path.replace("/", "_"))
        os.makedirs(model_folder, exist_ok=True)
        
        result = run_evaluation(
            model_name_or_path=args.finetuned_path,
            num_eval_samples=args.num_samples,
            base_model_name=args.finetuned_base,
            batch_size=args.batch_size,
            output_folder=model_folder,
            data_dir=args.data_dir,
            dataset_name=None,
            eval_split=args.eval_split,
            compare_with_baseline=True,  # Compare with baseline
            baseline_model="unsloth/Qwen2.5-VL-3B-Instruct"  # Default baseline model
        )
        if result:
            all_results.append(result)
            # Add model identifier to each sample result
            for sample_result in result["results"]:
                all_sample_results.append({
                    "model_id": result["model_id"],
                    "model_type": result["eval_type"],
                    "sample_id": sample_result["sample_id"],
                    "similarity": sample_result["similarity"],
                    "exact_match": sample_result["exact_match"]
                })

    # Create consolidated results CSV
    if all_sample_results:
        results_df = pd.DataFrame(all_sample_results)
        results_df.to_csv(os.path.join(run_folder, "all_results.csv"), index=False)
        
        # Create summary table
        summary_data = []
        for r in all_results:
            exact_match_count = sum(1 for res in r["results"] if res["exact_match"])
            summary_data.append({
                "Model": r["model_id"],
                "Type": r["eval_type"],
                "Avg Similarity": r["average_similarity"],
                "Exact Matches": f"{exact_match_count}/{r['num_samples']} ({exact_match_count/r['num_samples']*100:.1f}%)"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(run_folder, "summary.csv"), index=False)
        
        # Create a text summary
        with open(os.path.join(run_folder, "summary.txt"), "w") as f:
            f.write(f"EVALUATION RUN: {run_id}\n")
            f.write("=======================\n\n")
            for i, r in enumerate(all_results):
                f.write(f"Model {i+1}: {r['model_id']}\n")
                f.write(f"  Type: {r['eval_type']}\n")
                f.write(f"  Average Similarity: {r['average_similarity']:.4f}\n")
                exact_match_count = sum(1 for res in r["results"] if res["exact_match"])
                f.write(f"  Exact Matches: {exact_match_count}/{r['num_samples']} ({exact_match_count/r['num_samples']*100:.1f}%)\n\n")
            
            # Determine best model
            best_idx = np.argmax([r["average_similarity"] for r in all_results])
            f.write(f"Best performing model: {all_results[best_idx]['model_id']}\n")
            f.write(f"  Average Similarity: {all_results[best_idx]['average_similarity']:.4f}\n")
    
    # Create consolidated graphs
    if len(all_results) > 0:
        logger.info("Creating consolidated graphs...")
        
        # GRAPH 1: Per-model average similarity
        plt.figure(figsize=(14, 8))
        model_names = [r["model_id"] for r in all_results]
        avg_similarities = [r["average_similarity"] for r in all_results]
        
        bars = plt.bar(range(len(model_names)), avg_similarities, 
                      color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:len(model_names)])
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Average Similarity Score', fontsize=12)
        plt.title(f'Model Performance Comparison (Run {run_id})', fontsize=14, fontweight='bold')
        plt.xticks(range(len(model_names)), [f"{i+1}" for i in range(len(model_names))], rotation=0)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{avg_similarities[i]:.4f}', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
        
        # Add legend with model names
        plt.legend([f"{i+1}: {name}" for i, name in enumerate(model_names)], 
                  loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)
        
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, "model_comparison.png"), dpi=300)
        plt.close()
        
        # GRAPH 2: Per-sample comparison across all models
        if len(all_results) > 0:
            # Get unique sample IDs
            sample_ids = sorted(list(set(r["sample_id"] for r in all_sample_results)))
            
            # Create a DataFrame with samples as rows and models as columns
            comparison_data = {}
            for model_result in all_results:
                model_id = model_result["model_id"]
                model_samples = {r["sample_id"]: r["similarity"] for r in model_result["results"]}
                comparison_data[model_id] = [model_samples.get(sample_id, np.nan) for sample_id in sample_ids]
            
            comparison_df = pd.DataFrame(comparison_data, index=sample_ids)
            
            # Plot per-sample comparison
            plt.figure(figsize=(16, 10))
            
            # Plot each model's performance as a line
            for i, model_id in enumerate(comparison_df.columns):
                plt.plot(range(len(sample_ids)), comparison_df[model_id], 
                        marker='o', linestyle='-', linewidth=2, markersize=8,
                        label=f"Model {i+1}: {model_id}")
            
            plt.xlabel('Sample ID', fontsize=12)
            plt.ylabel('Similarity Score', fontsize=12)
            plt.title(f'Per-Sample Performance Comparison (Run {run_id})', fontsize=14, fontweight='bold')
            plt.xticks(range(len(sample_ids)), [f"{i+1}" for i in range(len(sample_ids))], rotation=45)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)
            plt.ylim(0, 1.05)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "sample_comparison.png"), dpi=300)
            plt.close()
            
            # Also create a heatmap for better visualization
            plt.figure(figsize=(12, 8))
            sns.heatmap(comparison_df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, 
                       linewidths=.5, fmt=".2f", cbar_kws={"label": "Similarity Score"})
            plt.title(f'Similarity Score Heatmap (Run {run_id})', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(run_folder, "similarity_heatmap.png"), dpi=300)
            plt.close()
    
    logger.info(f"Evaluation complete. All results saved to {run_folder}")
