import json
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import argparse
import os

def load_stats(file_path):
    elapsed_seconds = []
    total_pps = []
    max_pps = []
    total_pages = 0
    num_samples = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            num_samples += 1
            data = json.loads(line)
            elapsed_seconds.append(data['elapsed_seconds'])
            total_pps.append(data['total_pages_per_second'])
            max_pps.append(data['max_pages_per_second'])
            total_pages = data['total_pages']  # Will keep the last value
    
    return elapsed_seconds, total_pps, max_pps, total_pages, num_samples

def create_graphs(run_path):
    # Get the stats file path and create output path
    stats_file = os.path.join(run_path, 'stats.jsonl')
    output_dir = os.path.join(run_path, 'graphs')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'graph.png')
    
    # Use dark style
    plt.style.use("dark_background")
    
    # Create single figure with dark gray background
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1E1E1E')  # Dark gray background
    ax.set_facecolor('#1E1E1E')
    
    # Load data
    elapsed_seconds, total_pps, _, total_pages, num_samples = load_stats(stats_file)
    
    # Set titles with additional information
    fig.suptitle('Load Test Performance', fontsize=14, color='#E0E0E0')  # Light gray text
    subtitle = f'Total Pages: {total_pages:,} | Samples: {num_samples}'
    ax.set_title(subtitle, color='#E0E0E0')
    
    # Plot Pages per Second with a nice teal color
    ax.plot(elapsed_seconds, total_pps, color='#00B4D8', linewidth=2, label='Pages/sec')
    
    # Mark maximum value with a contrasting coral color
    max_value = max(total_pps)
    max_idx = total_pps.index(max_value)
    max_time = elapsed_seconds[max_idx]
    ax.plot(max_time, max_value, 'o', color='#FF6B6B', label=f'Max: {max_value:.1f}', markersize=8)
    
    ax.set_xlabel('Elapsed Time (seconds)', color='#E0E0E0')
    ax.set_ylabel('Pages per Second', color='#E0E0E0')
    ax.grid(True, alpha=0.1, color='#666666')  # Subtle grid
    ax.legend(facecolor='#1E1E1E', edgecolor='#666666')
    
    # Style the axis lines and ticks
    ax.spines['bottom'].set_color('#666666')
    ax.spines['top'].set_color('#666666') 
    ax.spines['right'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    ax.tick_params(colors='#E0E0E0')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save the graph
    plt.savefig(output_file, facecolor='#1E1E1E', edgecolor='none', bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate performance graphs from load test stats')
    parser.add_argument('--input', required=True, help='Path to the run directory (e.g., runs/20250204_134614_795)')
    
    args = parser.parse_args()
    
    # Ensure the run path exists
    if not os.path.exists(args.input):
        print(f"Error: Run path '{args.input}' does not exist")
        return
    
    # Generate the graphs
    create_graphs(args.input)
    print(f"Graphs generated successfully in {args.input}/graph.png")

if __name__ == "__main__":
    main()
