import gradio as gr
import json
import os
import pandas as pd
from PIL import Image
import glob
import argparse

# --- Configuration ---
DEFAULT_BASE_OUTPUT_DIR = "output/"
DEFAULT_IMAGE_DIR_PATTERN = "data/*/*" # Pattern to find potential image dirs

# --- Helper Functions ---

def find_run_dirs(base_output_dir):
    """Finds directories containing evaluation_results/eval_scores.json"""
    run_dirs = []
    if not os.path.isdir(base_output_dir):
        print(f"Warning: Base output directory not found: {base_output_dir}")
        return []
    for potential_run_dir in glob.glob(os.path.join(base_output_dir, "*")):
        if os.path.isdir(potential_run_dir):
            results_file = os.path.join(potential_run_dir, "evaluation_results", "eval_scores.json")
            if os.path.exists(results_file):
                run_dirs.append(potential_run_dir)
    return sorted(run_dirs, reverse=True) # Show newest first

def find_image_dir(run_dir, image_base_dir_pattern):
    """Tries to find the corresponding image directory (heuristic)."""
    # This is a simple heuristic. A more robust way would be to store
    # the image_dir path used during evaluation within the results json or config.
    potential_image_dirs = glob.glob(image_base_dir_pattern)
    # Try to find a dir name that matches part of the run_dir name or data source name
    # This needs refinement based on your actual directory structure.
    # For now, just return the first one found or None.
    if potential_image_dirs:
        return potential_image_dirs[0] # Simplistic guess
    return None

def load_results(run_dir):
    """Loads eval_scores.json from a run directory."""
    results_file = os.path.join(run_dir, "evaluation_results", "eval_scores.json")
    if not os.path.exists(results_file):
        return None, f"eval_scores.json not found in {run_dir}"
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Error decoding JSON: {e}"
    except Exception as e:
        return None, f"Error loading results: {e}"

def get_sample_data(results_data, index, image_dir):
    """Extracts data for a specific sample index."""
    if not results_data or "individual_scores" not in results_data or index >= len(results_data["individual_scores"]):
        # Return default empty values if index is out of bounds or data is missing
        return None, "N/A", "N/A", "N/A", "N/A"

    sample = results_data["individual_scores"][index]
    image_name = sample.get("image", "N/A")
    similarity = sample.get("similarity", "N/A")
    # exact_match = sample.get("exact_match", "N/A") # Removed
    generated_html = sample.get("generated", "")
    reference_html = sample.get("reference", "")

    # Attempt to load image
    image_path = None
    img = None
    if image_dir and image_name != "N/A":
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
                img = None # Set to None if loading fails
        else:
             print(f"Warning: Image path not found: {image_path}")


    # Clean HTML for display (remove ```html blocks if present)
    generated_html = generated_html.replace("```html", "").replace("```", "").strip()
    reference_html = reference_html.replace("```html", "").replace("```", "").strip()

    # Return data without exact_match
    return img, image_name, similarity, generated_html, reference_html

# --- Gradio Interface Logic ---

def update_run_selection(run_dir, image_base_dir_pattern):
    """Called when a run directory is selected."""
    results_data, error = load_results(run_dir)
    image_dir = find_image_dir(run_dir, image_base_dir_pattern) # Find image dir for this run
    status = f"Loaded: {os.path.basename(run_dir)}"
    num_samples = 0
    initial_img, initial_name, initial_sim, initial_gen, initial_ref = None, "N/A", "N/A", "", ""

    if error:
        print(error)
        status = f"Error: {error}"
        results_data = None # Ensure data is None on error
        image_dir = None
    elif results_data and "individual_scores" in results_data:
        num_samples = len(results_data["individual_scores"])
        if num_samples > 0:
            # Load data for the first sample (index 0)
            initial_img, initial_name, initial_sim, initial_gen, initial_ref = get_sample_data(results_data, 0, image_dir)
        else:
            status += " (No samples found)"
    else:
        status += " (No individual_scores found)"
        results_data = None # Ensure data is None if scores are missing
        image_dir = None

    # Update UI elements: Number input, status, initial sample view, and image dir state
    # Note: We return updates for the Number input's properties (value, maximum, interactive, visible)
    # and the initial values for the sample display components.
    return (
        results_data,
        gr.Number(value=0, maximum=max(0, num_samples - 1), interactive=num_samples > 0, visible=num_samples > 0), # Update Number input
        status,
        initial_img, initial_name, initial_sim, initial_gen, initial_ref,
        image_dir # Pass the found image directory to the state
    )


def update_sample_view(results_data, index, image_dir):
    """Called when the sample index changes."""
    # Ensure index is an integer
    try:
        index = int(index)
    except (ValueError, TypeError):
        index = 0 # Default to 0 if index is invalid

    if not results_data or "individual_scores" not in results_data:
        return None, "N/A", "N/A", "", "" # Return empty if no data

    num_samples = len(results_data["individual_scores"])
    if not (0 <= index < num_samples):
         # Handle out-of-bounds index gracefully, maybe show the first/last sample or empty
         index = max(0, min(index, num_samples - 1)) # Clamp index to valid range
         if num_samples == 0:
              return None, "N/A", "N/A", "", "" # Return empty if no samples

    # Get data for the selected index
    img, image_name, similarity, generated_html, reference_html = get_sample_data(results_data, index, image_dir)

    # Return updated values for the display components
    return img, image_name, similarity, generated_html, reference_html

def change_sample_index(current_index, change, results_data):
    """Handles Previous/Next button clicks."""
    if not results_data or "individual_scores" not in results_data:
        return 0 # Default to 0 if no data
    num_samples = len(results_data["individual_scores"])
    if num_samples == 0:
        return 0

    try:
        current_index = int(current_index)
    except (ValueError, TypeError):
        current_index = 0

    new_index = current_index + change
    # Clamp the index within the valid range [0, num_samples - 1]
    new_index = max(0, min(new_index, num_samples - 1))
    return new_index


# --- Build Gradio App ---

def build_app(base_output_dir, initial_run_dir, image_base_dir_pattern):
    """Builds the Gradio interface."""
    run_dirs = find_run_dirs(base_output_dir)

    if not run_dirs:
        print(f"Error: No valid evaluation runs found in {base_output_dir}.")
        print("Please ensure directories exist with 'evaluation_results/eval_scores.json' inside.")
        # Optionally, launch Gradio with an error message
        with gr.Blocks() as demo:
            gr.Markdown(f"# Error \n No valid evaluation runs found in `{base_output_dir}`.")
        return demo

    initial_run = initial_run_dir if initial_run_dir in run_dirs else run_dirs[0]

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Evaluation Results Viewer")

        # Hidden state to store loaded results and image dir for the selected run
        results_data_state = gr.State(None)
        image_dir_state = gr.State(None)

        with gr.Row():
            run_selector = gr.Dropdown(
                choices=run_dirs,
                value=initial_run,
                label="Select Evaluation Run",
                interactive=True,
                scale=3 # Make dropdown wider
            )
            status_display = gr.Textbox(label="Status", interactive=False, scale=1)

        # Sample Navigation Row
        with gr.Row(visible=False) as navigation_row: # Initially hidden
             with gr.Column(scale=1, min_width=50): # Dummy column for spacing
                 pass
             with gr.Column(scale=2, min_width=300): # Controls centered
                 with gr.Row():
                    prev_button = gr.Button("⬅️ Previous")
                    sample_index_input = gr.Number(
                        label="Sample Index",
                        minimum=0,
                        maximum=0, # Will be updated
                        value=0,
                        step=1,
                        interactive=False, # Initially disabled
                        # visible=False # Controlled by parent row
                    )
                    next_button = gr.Button("Next ➡️")
             with gr.Column(scale=1, min_width=50): # Dummy column for spacing
                 pass


        # Main Display Area
        with gr.Row():
            # Left Column: Image and minimal info
            with gr.Column(scale=1):
                image_display = gr.Image(label="Input Image", type="pil")
                with gr.Row(): # Put filename and similarity side-by-side
                    image_name_display = gr.Textbox(label="Filename", interactive=False, scale=2)
                    similarity_display = gr.Textbox(label="Similarity", interactive=False, scale=1)
                # exact_match_display = gr.Textbox(label="Exact Match", interactive=False) # Removed

            # Right Columns: HTML Tables
            with gr.Column(scale=2):
                generated_html = gr.HTML(label="Generated Table")
            with gr.Column(scale=2):
                reference_html = gr.HTML(label="Reference Table")

        # --- Interactions ---

        # When a run is selected:
        # - Load its data (results_data_state, image_dir_state)
        # - Update the status text (status_display)
        # - Update the sample index input (sample_index_input: value, max, interactive, visibility via parent row)
        # - Display the first sample (image_display, image_name_display, similarity_display, generated_html, reference_html)
        run_selector.change(
            fn=update_run_selection,
            inputs=[run_selector, gr.State(image_base_dir_pattern)],
            outputs=[
                results_data_state, sample_index_input, status_display,
                image_display, image_name_display, similarity_display,
                generated_html, reference_html,
                image_dir_state
            ]
        ).then(
            # Make navigation visible only if samples exist
            lambda data: gr.Row(visible=bool(data and data.get("individual_scores"))),
            inputs=[results_data_state],
            outputs=[navigation_row]
        )


        # When the sample index input changes (directly or via buttons):
        # - Update the displayed sample details
        sample_index_input.change(
            fn=update_sample_view,
            inputs=[results_data_state, sample_index_input, image_dir_state],
            outputs=[
                image_display, image_name_display, similarity_display,
                generated_html, reference_html
            ]
        )

        # Previous button click:
        # - Calculate new index
        # - Update the sample index input number
        prev_button.click(
            fn=change_sample_index,
            inputs=[sample_index_input, gr.State(-1), results_data_state], # Pass -1 for change
            outputs=[sample_index_input] # This automatically triggers sample_index_input.change
        )

        # Next button click:
        # - Calculate new index
        # - Update the sample index input number
        next_button.click(
            fn=change_sample_index,
            inputs=[sample_index_input, gr.State(1), results_data_state], # Pass 1 for change
            outputs=[sample_index_input] # This automatically triggers sample_index_input.change
        )


        # Trigger initial load for the default run when the app starts
        demo.load(
             fn=update_run_selection,
             inputs=[run_selector, gr.State(image_base_dir_pattern)],
             outputs=[
                 results_data_state, sample_index_input, status_display,
                 image_display, image_name_display, similarity_display,
                 generated_html, reference_html,
                 image_dir_state
             ]
        ).then(
            # Make navigation visible only if samples exist on initial load
            lambda data: gr.Row(visible=bool(data and data.get("individual_scores"))),
            inputs=[results_data_state],
            outputs=[navigation_row]
        )


    return demo

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio App for Viewing Evaluation Results")
    parser.add_argument("--base_output_dir", type=str, default=DEFAULT_BASE_OUTPUT_DIR, help="Base directory containing run output folders.")
    parser.add_argument("--initial_run_dir", type=str, default=None, help="Specific run directory to select initially.")
    parser.add_argument("--image_base_dir_pattern", type=str, default=DEFAULT_IMAGE_DIR_PATTERN, help="Glob pattern to search for image directories.")
    parser.add_argument("--share", action="store_true", help="Create a publicly shareable link.")
    parser.add_argument("--server_port", type=int, default=7860, help="Port to run the Gradio server on.")

    args = parser.parse_args()

    app = build_app(args.base_output_dir, args.initial_run_dir, args.image_base_dir_pattern)
    app.launch(share=args.share, server_name="0.0.0.0", server_port=args.server_port) 