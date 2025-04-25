import os
import re
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import matplotlib.pyplot as plt
from extractor import EXTRACTOR_KEYS
from processors import MODELS

# Reuse the color mapping function
def create_model_color_mapping():
    all_colors = []
    for palette in [px.colors.qualitative.Plotly, px.colors.qualitative.D3, 
                   px.colors.qualitative.G10, px.colors.qualitative.T10, 
                   px.colors.qualitative.Alphabet]:
        all_colors.extend(palette)
    
    color_mapping = {}
    for i, extractor_key in enumerate(EXTRACTOR_KEYS):
        color_index = i % len(all_colors)
        color_mapping[extractor_key] = all_colors[color_index]
    
    return color_mapping

MODEL_COLORS = create_model_color_mapping()

def load_latest_benchmark_run(runs_dir="runs"):
    """Load the most recent benchmark run data"""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"Runs directory {runs_dir} not found.")
        return None
    
    run_dirs = [d for d in os.listdir(runs_path) if (runs_path / d).is_dir()]
    if not run_dirs:
        print(f"No run directories found in {runs_path}")
        return None
    
    # Find the most recent run with results
    latest_run = None
    latest_time = 0
    
    for run_id in run_dirs:
        run_path = runs_path / run_id
        results_file = run_path / "scoring_results" / "results.jsonl"
        
        if results_file.exists():
            file_time = os.path.getmtime(results_file)
            if file_time > latest_time:
                latest_time = file_time
                latest_run = run_id
    
    if not latest_run:
        print("No valid benchmark runs found.")
        return None
    
    # Load the data from the latest run
    results_file = runs_path / latest_run / "scoring_results" / "results.jsonl"
    data = []
    
    try:
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        return {
            "run_id": latest_run,
            "data": data
        }
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        return None

def create_processor_chart(run_data, output_dir="bnlogs"):
    """Create a chart showing average score by processor and save it as an image"""
    if not run_data or not run_data.get("data"):
        print("No benchmark data available")
        return None
    
    df = pd.DataFrame(run_data["data"])
    
    # Check if required columns exist
    if not all(col in df.columns for col in ['processor', 'score']):
        print("Missing required columns in benchmark data")
        return None
    
    # Group by processor to get average scores
    processor_scores = df.groupby('processor')['score'].mean().reset_index()
    
    # Count samples for each processor
    processor_counts = df.groupby('processor').size().reset_index(name='count')
    processor_scores = pd.merge(processor_scores, processor_counts, on='processor')
    
    # Sort by score in descending order
    processor_scores = processor_scores.sort_values('score', ascending=False)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create a matplotlib figure for the chart
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    bars = plt.bar(
        processor_scores['processor'],
        processor_scores['score'],
        color=[MODEL_COLORS.get(p, '#1f77b4') for p in processor_scores['processor']]
    )
    
    # Add score labels on top of bars
    for bar, score, count in zip(bars, processor_scores['score'], processor_scores['count']):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{score:.2f} (n={count})',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Customize the chart
    plt.title('Average Score by Processor', fontsize=16, pad=20)
    plt.xlabel('Processor', fontsize=14, labelpad=10)
    plt.ylabel('Average Score', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, max(processor_scores['score']) * 1.15)  # Add some space for labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the chart as an image
    chart_path = output_path / "processor_scores.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to {chart_path}")
    return chart_path

def embed_chart_in_markdown(md_file_path, chart_path):
    """Embed the chart in the markdown file at the Results section"""
    if not md_file_path.exists():
        print(f"Markdown file {md_file_path} not found.")
        return False
    
    if not chart_path or not chart_path.exists():
        print(f"Chart image {chart_path} not found.")
        return False
    
    # Read the markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the Results section
    results_pattern = r'(### Results\s*\n)'
    match = re.search(results_pattern, content)
    
    if not match:
        print("Results section not found in the markdown file.")
        return False
    
    # Create the image markdown
    chart_markdown = f"\n\n![Average Score by Processor]({chart_path.relative_to(md_file_path.parent)})\n\n"
    
    # Insert the chart after the Results heading
    new_content = content[:match.end()] + chart_markdown + content[match.end():]
    
    # Write the updated content back to the file
    with open(md_file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Chart embedded in {md_file_path}")
    return True

def update_blog_with_chart():
    """Main function to update the blog with the processor score chart"""
    # Load the latest benchmark run
    run_data = load_latest_benchmark_run()
    
    if not run_data:
        print("No benchmark data to process")
        return
    
    # Create the chart
    chart_path = create_processor_chart(run_data)
    
    if not chart_path:
        print("Failed to create chart")
        return
    
    # Embed the chart in the markdown file
    md_file_path = Path("recent_llms.md")
    success = embed_chart_in_markdown(md_file_path, chart_path)
    
    if success:
        print(f"Successfully updated {md_file_path} with benchmark results chart")
    else:
        print(f"Failed to update {md_file_path}")

if __name__ == "__main__":
    update_blog_with_chart() 