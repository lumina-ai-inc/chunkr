import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tensorboard_dir", type=str, default=None, 
                        help="Path to tensorboard logs from training")
    return parser.parse_args()

def plot_tensorboard_metrics(tensorboard_dir, output_dir):
    """Extract and plot metrics from TensorBoard logs"""
    if not tensorboard_dir or not os.path.exists(tensorboard_dir):
        print("TensorBoard directory not found or not specified. Skipping training metrics visualization.")
        return
    
    try:
        # Import tensorboard required libraries
        from tensorboard.backend.event_processing import event_accumulator
        
        # Find all event files
        event_files = glob.glob(os.path.join(tensorboard_dir, "events.out.tfevents.*"))
        
        if not event_files:
            print("No TensorBoard event files found. Skipping training metrics visualization.")
            return
        
        # Create metrics plots directory
        metrics_dir = os.path.join(output_dir, "training_metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Process each event file
        for event_file in event_files:
            ea = event_accumulator.EventAccumulator(
                event_file,
                size_guidance={
                    event_accumulator.SCALARS: 0,
                }
            )
            ea.Reload()
            
            # Get all scalar tags (metrics)
            tags = ea.Tags()['scalars']
            
            # Plot each metric
            for tag in tags:
                events = ea.Scalars(tag)
                steps = [event.step for event in events]
                values = [event.value for event in events]
                
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(steps, values)
                plt.title(tag)
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.grid(True)
                
                # Save plot
                safe_tag = tag.replace('/', '_')
                plt.savefig(os.path.join(metrics_dir, f"{safe_tag}.png"))
                plt.close()
        
        print(f"Training metrics visualizations saved to {metrics_dir}")
    except Exception as e:
        print(f"Error processing TensorBoard logs: {e}")
        print("Continuing with other visualizations...")

def compute_nlp_metrics(results):
    """Compute simple text similarity metrics between ground truth and predictions"""
    try:
        # Try to import specialized NLP metrics
        from rouge_score import rouge_scorer
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Initialize metrics
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        
        # Initialize scorers
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoothing = SmoothingFunction().method1
        
        for result in results:
            # Get ground truth and model response
            ground_truth = result['ground_truth']
            model_response = result['model_response']
            
            # Compute ROUGE scores
            rouge_result = rouge_scorer_instance.score(ground_truth, model_response)
            rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
            
            # Compute BLEU score
            reference = nltk.word_tokenize(ground_truth.lower())
            candidate = nltk.word_tokenize(model_response.lower())
            try:
                bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)
                bleu_scores.append(bleu)
            except Exception:
                # Skip problematic examples
                pass
        
        # Calculate average scores
        avg_metrics = {
            'rouge1': np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0,
            'rouge2': np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0,
            'rougeL': np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0,
            'bleu': np.mean(bleu_scores) if bleu_scores else 0
        }
        
    except ImportError:
        # Fallback to simple token overlap if advanced metrics aren't available
        print("NLP metrics libraries not found. Using simple token overlap.")
        overlaps = []
        
        for result in results:
            gt_tokens = set(result['ground_truth'].lower().split())
            resp_tokens = set(result['model_response'].lower().split())
            
            if gt_tokens:
                overlap = len(gt_tokens.intersection(resp_tokens)) / len(gt_tokens)
                overlaps.append(overlap)
        
        avg_metrics = {
            'token_overlap': np.mean(overlaps) if overlaps else 0,
            'rouge1': 0,
            'rouge2': 0,
            'rougeL': 0,
            'bleu': 0
        }
    
    return avg_metrics

def visualize_examples(results, output_dir, num_examples=5):
    """Visualize a few example results"""
    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    # Select random examples
    indices = np.random.choice(len(results), min(num_examples, len(results)), replace=False)
    
    for i, idx in enumerate(indices):
        result = results[idx]
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # If there's an image, display it
        try:
            if isinstance(result['image_path'], list):
                image_path = result['image_path'][0]  # Just use the first image
            else:
                image_path = result['image_path']
                
            if os.path.exists(image_path):
                img = Image.open(image_path)
                ax.imshow(img)
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(examples_dir, f"example_{i}_image.png"))
                plt.close()
        except Exception as e:
            print(f"Error displaying image for example {i}: {e}")
            plt.close()
            continue
        
        # Create a text file with prompt, ground truth, and model response
        with open(os.path.join(examples_dir, f"example_{i}_text.txt"), 'w') as f:
            f.write(f"PROMPT:\n{result['prompt']}\n\n")
            f.write(f"GROUND TRUTH:\n{result['ground_truth']}\n\n")
            f.write(f"MODEL RESPONSE:\n{result['model_response']}\n")
    
    print(f"Example visualizations saved to {examples_dir}")

def create_summary_report(metrics, output_dir):
    """Create a summary report with metrics"""
    report_path = os.path.join(output_dir, "evaluation_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("# Model Evaluation Summary\n\n")
        f.write("## NLP Metrics\n\n")
        
        if 'token_overlap' in metrics:
            f.write(f"- Token Overlap: {metrics['token_overlap']:.4f}\n")
        
        f.write(f"- ROUGE-1: {metrics['rouge1']:.4f}\n")
        f.write(f"- ROUGE-2: {metrics['rouge2']:.4f}\n")
        f.write(f"- ROUGE-L: {metrics['rougeL']:.4f}\n")
        f.write(f"- BLEU: {metrics['bleu']:.4f}\n")
    
    print(f"Summary report saved to {report_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Only process TensorBoard logs if results file doesn't exist
    if not os.path.exists(args.results_path):
        print(f"Results file not found: {args.results_path}")
        print("Visualizing TensorBoard metrics only...")
        plot_tensorboard_metrics(args.tensorboard_dir, args.output_dir)
        
        # Create a placeholder report
        with open(os.path.join(args.output_dir, "status.txt"), 'w') as f:
            f.write("Evaluation results are not available.\n")
            f.write("Only TensorBoard metrics have been visualized.\n")
        
        return
    
    # Load results
    with open(args.results_path, 'r') as f:
        results = json.load(f)
    
    # Visualize training metrics from TensorBoard
    plot_tensorboard_metrics(args.tensorboard_dir, args.output_dir)
    
    # Compute NLP metrics
    metrics = compute_nlp_metrics(results)
    
    # Visualize examples
    visualize_examples(results, args.output_dir)
    
    # Create summary report
    create_summary_report(metrics, args.output_dir)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 