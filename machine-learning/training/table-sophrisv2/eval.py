import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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

def compare_html(generated_html, ground_truth_html):
    # Normalize HTML - remove whitespace and newlines for comparison
    gen_clean = generated_html.strip().replace("\n", "").replace(" ", "").lower()
    gt_clean = ground_truth_html.strip().replace("\n", "").replace(" ", "").lower()
    
    # Calculate similarity
    if not gen_clean or not gt_clean:
        return 0.0
    
    # If exact match
    if gen_clean == gt_clean:
        return 1.0
        
    # Otherwise use Levenshtein distance for similarity
    max_len = max(len(gen_clean), len(gt_clean))
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
    baseline_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
):
    # --- Setup for this evaluation run ---
    is_finetuned = base_model_name is not None
    model_id = model_name_or_path.split("/")[-1] if "/" in model_name_or_path else model_name_or_path
    eval_type = "Fine-tuned" if is_finetuned else "Baseline"
    
    # Create output folder
    run_folder = os.path.join(output_folder, f"{eval_type}_{model_id}")
    os.makedirs(run_folder, exist_ok=True)
    
    logger.info(f"{'='*20} Starting Evaluation for: {model_id} (Type: {eval_type}) {'='*20}")
    
    # Generation parameters
    temperature = 0.1
    min_p = 0.1
    max_new_tokens = 2048
    data_limit = num_eval_samples * 2

    # --- Load Model and Processor ---
    try:
        if is_finetuned:
            logger.info(f"Loading FINE-TUNED model from '{model_name_or_path}'")
            if not os.path.exists(model_name_or_path):
                logger.error(f"Model path not found: {model_name_or_path}")
                logger.error("Skipping evaluation for this fine-tuned model.")
                return
            
            # Load the model directly (not using LoRA adapters)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_name_or_path)
                
        else:
            logger.info(f"Loading BASE model: '{model_name_or_path}'")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        logger.info("Model loaded successfully.")
        model.eval()  # Set to evaluation mode
    except Exception as e:
        logger.error(f"Failed to load model '{model_id}': {e}")
        logger.error("Skipping evaluation for this model.")
        if 'model' in locals(): del model
        if 'processor' in locals(): del processor
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
            del processor
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
            
            try:
                inputs = processor(
                    text=processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
                    images=image,
                    return_tensors="pt"
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
                        **inputs, 
                        max_new_tokens=max_new_tokens, 
                        use_cache=True,
                        temperature=temperature, 
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
                    )
            except Exception as e:
                logger.error(f"Error during model generation for sample {sample.table_id}: {e}")
                continue

            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            generated_text_raw = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
        plt.xlabel('Sample ID', fontsize=12)
        plt.ylabel('Similarity Score', fontsize=12)
        plt.title(f'HTML Similarity Scores ({eval_type}: {model_id})', fontsize=14, fontweight='bold')
        plt.xticks(range(len(sample_ids)), [f"{i+1}" for i in range(len(sample_ids))], rotation=45)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, "similarity_scores.png"), dpi=300)
        plt.close()
        
        # Plot exact matches distribution
        plt.figure(figsize=(10, 6))
        exact_count = sum(exact_matches)
        non_exact_count = len(exact_matches) - exact_count
        plt.pie([exact_count, non_exact_count], 
                labels=['Exact Match', 'Partial Match'], 
                autopct='%1.1f%%',
                colors=['#4CAF50', '#FF9800'],
                explode=(0.1, 0),
                shadow=True)
        plt.title(f'Exact vs. Partial Matches ({eval_type}: {model_id})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, "exact_matches.png"), dpi=300)
        plt.close()
    
    logger.info(f"Results and graphs saved to: {run_folder}")
    return {
        "model_id": model_id,
        "eval_type": eval_type,
        "average_similarity": average_similarity,
        "num_samples": len(raw_samples),
        "results": results,
        "run_folder": run_folder
    }

def compare_models(model_results, run_id, output_folder):
    """
    Generate comparison charts between different models
    """
    if len(model_results) <= 1:
        logger.info("Not enough models to compare (need at least 2).")
        return
    
    run_folder = os.path.join(output_folder, f"comparison_{run_id}")
    os.makedirs(run_folder, exist_ok=True)
    
    logger.info(f"Generating model comparison visualizations in {run_folder}")
    
    # Extract model names and average scores
    model_names = [result["model_id"] for result in model_results]
    avg_scores = [result["average_similarity"] for result in model_results]
    eval_types = [result["eval_type"] for result in model_results]
    
    # Create DataFrame for combined model results
    comparison_df = pd.DataFrame({
        "Model": model_names,
        "Type": eval_types,
        "Average Similarity": avg_scores
    })
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(run_folder, "model_comparison.csv"), index=False)
    
    # Bar chart of average scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(model_names)), avg_scores, color=['skyblue' if t == 'Baseline' else 'orange' for t in eval_types])
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{avg_scores[i]:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Similarity Score', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    
    # Add legend for model types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Baseline'),
        Patch(facecolor='orange', label='Fine-tuned')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Per-sample comparison if we have consistent sample IDs across models
    all_sample_results = []
    for result in model_results:
        for sample_result in result["results"]:
            all_sample_results.append({
                "Model": result["model_id"],
                "Type": result["eval_type"],
                "Sample ID": sample_result["sample_id"],
                "Similarity": sample_result["similarity"]
            })
    
    sample_df = pd.DataFrame(all_sample_results)
    
    # Check if we have the same samples across models
    sample_counts = sample_df.groupby("Sample ID").size()
    if (sample_counts == len(model_results)).all():
        # We have consistent samples - create a comparison chart
        sample_ids = sorted(sample_df["Sample ID"].unique())
        
        # Create a pivot table for the data
        comparison_df = pd.pivot_table(
            sample_df, 
            values='Similarity', 
            index='Sample ID', 
            columns='Model',
            aggfunc='first'
        )
        
        # Plot the comparison
        plt.figure(figsize=(14, 8))
        for model in comparison_df.columns:
            model_type = sample_df[sample_df["Model"] == model]["Type"].iloc[0]
            linestyle = '-' if model_type == 'Baseline' else '--'
            marker = 'o' if model_type == 'Baseline' else '^'
            plt.plot(comparison_df.index, comparison_df[model], label=model, 
                    linestyle=linestyle, marker=marker, linewidth=2, markersize=6)
            
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

if __name__ == "__main__":
    load_dotenv(override=True)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate table OCR models")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate per model")
    parser.add_argument("--baseline_models", nargs="+", default=["Qwen/Qwen2.5-VL-3B-Instruct"], 
                        help="List of baseline models to evaluate")
    parser.add_argument("--finetuned_path", type=str, default="outputs/qwen2_5_Vl_3B-table-finetune", 
                        help="Path to finetuned model directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", 
                        help="Base model for the finetuned model")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline model evaluation")
    parser.add_argument("--skip_finetuned", action="store_true", help="Skip finetuned model evaluation")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory containing train/val/test split data")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of the dataset folder within data_dir")
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
    
    # Evaluate baseline models
    if not args.skip_baseline:
        for model_name in args.baseline_models:
            logger.info(f"Evaluating baseline model: {model_name}")
            try:
                result = run_evaluation(
                    model_name_or_path=model_name,
                    num_eval_samples=args.num_samples,
                    batch_size=args.batch_size,
                    output_folder=run_folder,
                    data_dir=args.data_dir,
                    dataset_name=args.dataset_name,
                    eval_split=args.eval_split
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating baseline model {model_name}: {e}")
    else:
        logger.info("Skipping baseline model evaluation as requested.")
            
    # Evaluate fine-tuned model if available
    if not args.skip_finetuned and args.finetuned_path:
        # Find the latest model directory if it's a nested structure
        if os.path.isdir(args.finetuned_path):
            model_dirs = [d for d in os.listdir(args.finetuned_path) 
                          if os.path.isdir(os.path.join(args.finetuned_path, d)) and d.startswith("model_")]
            if model_dirs:
                # Sort by modification time (newest first)
                model_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(args.finetuned_path, x)), reverse=True)
                finetuned_path = os.path.join(args.finetuned_path, model_dirs[0])
                logger.info(f"Found most recent model directory: {finetuned_path}")
            else:
                finetuned_path = args.finetuned_path
        else:
            finetuned_path = args.finetuned_path
            
        if not os.path.exists(finetuned_path):
            logger.error(f"Finetuned model path not found: {finetuned_path}")
        else:
            logger.info(f"Evaluating fine-tuned model at: {finetuned_path}")
            try:
                result = run_evaluation(
                    model_name_or_path=finetuned_path,
                    num_eval_samples=args.num_samples,
                    base_model_name=args.base_model,
                    batch_size=args.batch_size,
                    output_folder=run_folder,
                    data_dir=args.data_dir,
                    dataset_name=args.dataset_name,
                    eval_split=args.eval_split
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating fine-tuned model: {e}")
    else:
        logger.info("Skipping fine-tuned model evaluation as requested.")
    
    # Generate comparison if we have multiple models
    if len(all_results) > 1:
        compare_models(all_results, run_id, run_folder)
    
    logger.info(f"Evaluation complete! All results saved to {run_folder}")
