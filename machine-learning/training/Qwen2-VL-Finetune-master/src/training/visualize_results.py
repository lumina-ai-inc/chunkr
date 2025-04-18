import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
import glob
import base64
from io import BytesIO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing the evaluation images")
    parser.add_argument("--tensorboard_dir", type=str, default=None,
                        help="Path to tensorboard logs from training")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Hub model ID for uploading results")
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
            print(f"Processing TensorBoard events from: {event_file}")
            ea = event_accumulator.EventAccumulator(event_file,
                                                   size_guidance={
                                                       event_accumulator.SCALARS: 0,
                                                   })
            ea.Reload()
            
            # Get all scalar tags and visualize each one
            tags = ea.Tags()['scalars']
            for tag in tags:
                events = ea.Scalars(tag)
                steps = [event.step for event in events]
                values = [event.value for event in events]
                
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
        print(f"To view metrics in real-time, access TensorBoard at: http://<your-sf-compute-ip>:6006")
    except Exception as e:
        print(f"Error processing TensorBoard logs: {e}")
        print("Continuing with other visualizations...")

def compute_nlp_metrics(results):
    """Computes NLP metrics (BLEU, ROUGE, METEOR) from evaluation results."""
    try:
        # Lazy import evaluate only when needed
        import evaluate 
    except ImportError:
        print("Warning: 'evaluate' library not installed. Skipping NLP metrics.")
        print("Install it using: pip install evaluate rouge_score nltk")
        return {}

    # Initialize metrics - handle potential FileNotFoundError if cache is corrupted
    try:
        bleu = evaluate.load("bleu")
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
    except (FileNotFoundError, Exception) as e:
         print(f"Warning: Failed to load evaluation metrics: {e}. Skipping NLP metrics.")
         print("Consider clearing the huggingface cache: rm -rf ~/.cache/huggingface/evaluate")
         return {}

    predictions = []
    references = []

    if 'individual_scores' not in results or not isinstance(results['individual_scores'], list):
        print("Warning: 'individual_scores' key missing or not a list in results. Skipping NLP metrics.")
        return {} # Return empty dict if data is missing/malformed

    for result in results['individual_scores']:
        # Check if result is a dictionary and contains the required keys
        if isinstance(result, dict) and 'generated' in result and 'reference' in result:
            # Extract generated and reference HTML, clean them
            generated_html = clean_html(result.get('generated', '')) # Use .get for safety
            reference_html = clean_html(result.get('reference', '')) # Use .get for safety

            # Convert HTML to text for NLP metrics (simple approach)
            # A more sophisticated approach might parse the HTML better
            generated_text = re.sub('<[^<]+?>', '', generated_html).strip()
            reference_text = re.sub('<[^<]+?>', '', reference_html).strip()

            predictions.append(generated_text)
            # ROUGE/METEOR expect a list of references for each prediction
            references.append([reference_text]) 
        else:
            print(f"Warning: Skipping malformed result item: {result}")
            # Optionally append empty strings or handle differently
            # predictions.append("")
            # references.append([""])

    if not predictions or not references:
        print("Warning: No valid predictions/references found to compute NLP metrics.")
        return {}

    # Compute metrics
    try:
        bleu_score = bleu.compute(predictions=predictions, references=references)
        rouge_score = rouge.compute(predictions=predictions, references=references)
        meteor_score = meteor.compute(predictions=predictions, references=references)
    except Exception as e:
        print(f"Error computing NLP metrics: {e}")
        return {}

    # Combine scores into a single dictionary
    metrics = {
        "bleu": bleu_score.get('bleu', 0.0), # Extract specific score
        "rouge1": rouge_score.get('rouge1', 0.0),
        "rouge2": rouge_score.get('rouge2', 0.0),
        "rougeL": rouge_score.get('rougeL', 0.0),
        "meteor": meteor_score.get('meteor', 0.0)
    }
    return metrics

def clean_html(html_string):
    """Removes markdown code fences from HTML strings."""
    if html_string.startswith("```html"):
        html_string = html_string[len("```html"):].strip()
    if html_string.endswith("```"):
        html_string = html_string[:-len("```")].strip()
    return html_string

def image_to_base64(image_path, max_size=(300, 300)):
    """Converts an image file to a base64 string for embedding in HTML."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            buffered = BytesIO()
            # Determine image format, default to PNG if unknown
            img_format = img.format if img.format else 'PNG'
            if img_format == 'JPEG':
                # Handle potential issues with JPEG saving (e.g., RGBA)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
            img.save(buffered, format=img_format)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/{img_format.lower()};base64,{img_str}"
    except Exception as e:
        print(f"Warning: Could not process image {image_path}: {e}")
        return None

def create_html_viewer(results_data, output_dir, image_dir):
    """Creates an HTML file to visualize evaluation results side-by-side."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Results Viewer</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .result-item {
            border: 1px solid #ccc;
            margin-bottom: 20px;
            padding: 15px;
            display: flex;
            flex-wrap: wrap; /* Allow wrapping on smaller screens */
            gap: 15px;
            align-items: flex-start; /* Align items at the top */
        }
        .image-container {
            flex: 1 1 300px; /* Flex-grow, flex-shrink, flex-basis */
            min-width: 250px;
            text-align: center;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #eee;
        }
        .table-container {
            flex: 2 1 400px; /* Allow tables to take more space */
            min-width: 350px;
            overflow-x: auto; /* Add horizontal scroll if needed */
        }
        .table-container table {
            border-collapse: collapse;
            width: 100%; /* Make tables take full width of container */
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .table-container th, .table-container td {
            border: 1px solid #ddd;
            padding: 6px;
            text-align: left;
        }
        .table-container th {
            background-color: #f2f2f2;
        }
        h3 { margin-top: 0; }
        .score { font-weight: bold; margin-top: 5px; }
        .filename { font-style: italic; color: #555; margin-bottom: 5px; word-break: break-all; }
    </style>
</head>
<body>
    <h1>Evaluation Results</h1>
"""

    html_content += f"<h2>Average Similarity: {results_data.get('average_similarity', 'N/A'):.4f}</h2>"
    html_content += f"<h3>Total Samples: {results_data.get('num_samples_evaluated', 'N/A')}</h3><hr>"

    for item in tqdm(results_data.get("individual_scores", []), desc="Generating HTML viewer"):
        image_name = item.get("image", "unknown.jpg")
        image_path = os.path.join(image_dir, image_name)
        generated_html = clean_html(item.get("generated", ""))
        reference_html = clean_html(item.get("reference", ""))
        similarity = item.get("similarity", "N/A")
        exact_match = item.get("exact_match", False)

        # Embed image as base64
        base64_image = image_to_base64(image_path)

        html_content += '<div class="result-item">\n'

        # Image Column
        html_content += '  <div class="image-container">\n'
        html_content += f'    <div class="filename">{image_name}</div>\n'
        if base64_image:
            html_content += f'    <img src="{base64_image}" alt="{image_name}">\n'
        else:
             html_content += f'    <p>Image not found or could not be loaded.</p>\n'
        html_content += f'    <div class="score">Similarity: {similarity:.4f}</div>\n'
        html_content += f'    <div class="score">Exact Match: {exact_match}</div>\n'
        html_content += '  </div>\n'

        # Generated Table Column
        html_content += '  <div class="table-container">\n'
        html_content += '    <h3>Generated Table</h3>\n'
        html_content += f'    {generated_html}\n'
        html_content += '  </div>\n'

        # Reference Table Column
        html_content += '  <div class="table-container">\n'
        html_content += '    <h3>Reference Table</h3>\n'
        html_content += f'    {reference_html}\n'
        html_content += '  </div>\n'

        html_content += '</div>\n' # Close result-item

    html_content += """
</body>
</html>
"""
    output_path = os.path.join(output_dir, "evaluation_viewer.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML viewer saved to {output_path}")

def create_summary_report(metrics, output_dir):
    """Create a summary report with metrics"""
    report_path = os.path.join(output_dir, "evaluation_summary.txt")

    with open(report_path, 'w') as f:
        f.write("# Model Evaluation Summary\n\n")

        # Check if any NLP metrics were computed
        if not metrics:
            f.write("NLP metrics could not be computed (likely missing 'evaluate' library).\n")
            f.write("Install with: pip install evaluate rouge_score nltk\n")
            print(f"Summary report (without NLP metrics) saved to {report_path}")
            return # Exit the function early

        f.write("## NLP Metrics\n\n")

        # Check for each specific metric before writing
        if 'bleu' in metrics:
            f.write(f"- BLEU: {metrics['bleu']:.4f}\n")
        if 'rouge1' in metrics:
            f.write(f"- ROUGE-1: {metrics['rouge1']:.4f}\n")
        if 'rouge2' in metrics:
            f.write(f"- ROUGE-2: {metrics['rouge2']:.4f}\n")
        if 'rougeL' in metrics:
            f.write(f"- ROUGE-L: {metrics['rougeL']:.4f}\n")
        if 'meteor' in metrics:
            f.write(f"- METEOR: {metrics['meteor']:.4f}\n")
        # Add checks for other metrics if you compute them later

    print(f"Summary report saved to {report_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if results file exists before proceeding with full visualization
    results_file_path = args.results_path
    if not os.path.exists(results_file_path):
        print(f"Results file not found: {results_file_path}")
        print("Attempting to visualize TensorBoard metrics only...")
        plot_tensorboard_metrics(args.tensorboard_dir, args.output_dir)

        # Create a placeholder status report
        status_path = os.path.join(args.output_dir, "status.txt")
        with open(status_path, 'w') as f:
            f.write("Evaluation results file not found.\n")
            if args.tensorboard_dir and os.path.exists(args.tensorboard_dir):
                 f.write("Only TensorBoard metrics have been visualized.\n")
            else:
                 f.write("No visualizations could be generated.\n")
        print(f"Status report saved to {status_path}")
        return # Exit if no results file

    # Load results since the file exists
    print(f"Loading results from {results_file_path}...")
    try:
        with open(results_file_path, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {results_file_path}: {e}")
        # Create a status report indicating the error
        status_path = os.path.join(args.output_dir, "status.txt")
        with open(status_path, 'w') as f:
            f.write(f"Error loading evaluation results from {results_file_path}.\n")
            f.write(f"JSONDecodeError: {e}\n")
            f.write("Cannot generate detailed visualizations or metrics.\n")
        print(f"Status report saved to {status_path}")
        # Attempt to plot tensorboard metrics if available
        plot_tensorboard_metrics(args.tensorboard_dir, args.output_dir)
        return # Exit due to JSON error
    except Exception as e:
        print(f"An unexpected error occurred while loading {results_file_path}: {e}")
        # Create a status report indicating the error
        status_path = os.path.join(args.output_dir, "status.txt")
        with open(status_path, 'w') as f:
            f.write(f"Unexpected error loading evaluation results from {results_file_path}.\n")
            f.write(f"Error: {e}\n")
            f.write("Cannot generate detailed visualizations or metrics.\n")
        print(f"Status report saved to {status_path}")
        # Attempt to plot tensorboard metrics if available
        plot_tensorboard_metrics(args.tensorboard_dir, args.output_dir)
        return # Exit due to other loading error

    # Visualize training metrics from TensorBoard
    plot_tensorboard_metrics(args.tensorboard_dir, args.output_dir)
    
    # Compute NLP metrics
    metrics = compute_nlp_metrics(results)
    
    # Create the HTML viewer
    create_html_viewer(results, args.output_dir, args.image_dir)
    
    # Create summary report
    create_summary_report(metrics, args.output_dir)
    
    print("Local visualization complete!")

    # Upload the entire visualization output directory to Hugging Face Hub
    # ... existing code ...

if __name__ == "__main__":
    main() 