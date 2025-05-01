import gradio as gr
from PIL import Image
import json
import os
import re
import traceback
import openai
import base64
from io import BytesIO
import time
import Levenshtein

# --- Configuration ---
# Models & paths - Removed model loading specific configs
# BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" # Removed
# TRAINING_OUTPUT_DIR = "output/sophris_table_prod1" # Removed
# HUB_MODEL_ID_FALLBACK = "ChunkrAI/sophris-table-VLM" # Removed
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Removed
# MODEL_CACHE_DIR = "./hf_cache" # Removed

# vLLM Server details
VLLM_HOST = "localhost"
VLLM_PORT = 8001
VLLM_API_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1" # Standard vLLM OpenAI endpoint path
# Note: The specific model running on the vLLM server is determined
# when the vLLM server is launched, not by this script.
# We still need a placeholder model name for the API call,
# even if vLLM ignores it. Use the expected model name.
VLLM_EXPECTED_MODEL_NAME = "ChunkrAI/chunkr-table-v1-qwen2_5VL-7B" # Or your fine-tuned model name if served directly

# Data configuration
DATA_DIR_BASE = "data/sophris-datasheet-table-extraction-azure-distill-v1"
IMAGE_DIR = "data/sophris-datasheet-table-extraction-azure-distill-v1/sophris-datasheet-table-extraction-azure-distill-v1"
# DATASET_BASENAME = os.path.basename(DATA_DIR_BASE) # No longer needed for image path
# IMAGE_DIR = os.path.join(DATA_DIR_BASE, DATASET_BASENAME) # Original path construction
TEST_DATA_PATH = "data/llava-format/test.json"
NUM_SAMPLES = 1000
MAX_NEW_TOKENS = 1024

# --- Initialize OpenAI client ---
try:
    client = openai.OpenAI(
        base_url=VLLM_API_BASE_URL,
        api_key="dummy" # API key is often ignored by local vLLM servers
    )
    # Optional: Add a check to see if the server is reachable
    # client.models.list() # This would attempt a simple API call
    print(f"OpenAI client initialized for vLLM server at: {VLLM_API_BASE_URL}")
except openai.APIConnectionError as e:
    print(f"FATAL: Could not connect to vLLM OpenAI server at {VLLM_API_BASE_URL}")
    print(f"Error: {e}")
    print("Please ensure the vLLM server is running and accessible.")
    exit(1)
except Exception as e:
    print(f"FATAL: Error initializing OpenAI client: {e}")
    exit(1)

# --- Data Loading ---
def load_test_samples(data_path, image_base_dir, num_samples):
    """Loads samples from the test JSON file, including reference response."""
    print(f"Loading up to {num_samples} test samples from {data_path}...")
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {data_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_path}")
        return []

    samples = []
    for i, item in enumerate(data):
        if len(samples) >= num_samples:
            print(f"Reached target of {num_samples} valid samples after checking {i} entries.")
            break
        try:
            human_message = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'human'), '')
            human_message = human_message.replace("<image>", "").strip()
            reference_response = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt'), '')

            image_filename = item.get("image")
            if not image_filename or not isinstance(image_filename, str):
                continue

            image_path = os.path.join(image_base_dir, image_filename)

            if not os.path.exists(image_path):
                continue

            samples.append({
                "id": f"Sample {len(samples)+1} ({image_filename})",
                "prompt": human_message,
                "image_path": image_path,
                "image_filename": image_filename,
                "reference_response": reference_response
            })

        except Exception as e:
            print(f"Skipping entry {i+1} due to error: {e}")

    if len(samples) < num_samples:
        print(f"Warning: Found only {len(samples)} valid samples after checking all {len(data)} entries.")
    elif not samples:
        print("Warning: No valid samples could be loaded.")

    print(f"Loaded {len(samples)} samples.")
    return samples

# --- vLLM Inference ---
def pil_to_base64_data_uri(pil_img):
    """Converts a PIL Image to a base64 data URI."""
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def run_vllm_inference(prompt: str, image: Image.Image, max_new_tokens=MAX_NEW_TOKENS):
    """Runs inference using the vLLM OpenAI API."""
    if image is None:
        return "Inference Error: No image provided."

    try:
        image_url = pil_to_base64_data_uri(image)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]

        print(f"Sending request to vLLM API (model: {VLLM_EXPECTED_MODEL_NAME}), max_tokens={max_new_tokens}")

        completion = client.chat.completions.create(
            model=VLLM_EXPECTED_MODEL_NAME, # Use the expected model name (vLLM might ignore this)
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.2,
        )
        response = completion.choices[0].message.content
        print("vLLM response received.")
        return response.strip()

    except openai.APIConnectionError as e:
        error_msg = f"vLLM Connection Error: {e}"
        print(error_msg)
        return f"Inference Error: {error_msg}"
    except openai.APIStatusError as e:
        error_msg = f"vLLM API Error: Status {e.status_code}, Response: {e.response}"
        print(error_msg)
        return f"Inference Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return f"Inference Error: {error_msg}"

# --- HTML Parsing & Similarity ---
def extract_html(text):
    """Extracts HTML content from model output, handling various formats."""
    if not text or text.startswith("Inference Error:") or text.startswith("Processing Error:"):
            return None

    text_stripped = text.strip()

    # Try standard regex first (most common case)
    strict_match = re.search(r"```(?:html)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if strict_match:
        # print("Info: Found well-formed ```html``` block.")
        return strict_match.group(1).strip()

    # Helper to check if content starts like HTML
    def looks_like_html(content):
        content_lower_stripped = content.strip().lower()
        # Added more tags common at the start of table fragments
        return content_lower_stripped.startswith(("<table", "<tbody", "<thead", "<tr", "<td", "<th", "<div", "<html", "<p", "<ul", "<ol"))

    # Heuristic 1: Missing closing backticks
    start_match = re.search(r"```(?:html)?\s*", text, re.IGNORECASE)
    if start_match:
        potential_html = text[start_match.end():].strip()
        if looks_like_html(potential_html):
             # print("Info: Found opening ``` but no closing ```.")
             return potential_html

    # Heuristic 2: Missing opening backticks
    if text_stripped.endswith("```"):
         end_pos = text.rfind("```")
         if end_pos != -1:
             potential_html = text[:end_pos].strip()
             if looks_like_html(potential_html):
                 # print("Info: Found closing ``` but no opening ```.")
                 return potential_html

    # Heuristic 3: No backticks at all, but looks like HTML
    if looks_like_html(text_stripped):
        # print("Info: No ``` found, but text looks like HTML.")
        return text_stripped

    # Heuristic 4: Check if the *entire* string is likely HTML (e.g., starts with <table> ends with </table>)
    if text_stripped.startswith("<") and text_stripped.endswith(">"):
         # Basic check, could be refined
         if looks_like_html(text_stripped):
             # print("Info: Assuming entire string is HTML.")
             return text_stripped


    # print("Info: Could not extract HTML content.")
    return None # Return None if no HTML-like content found

def calculate_similarity(generated_html: str, reference_html: str) -> float:
    """Calculates Levenshtein similarity between two HTML strings after cleaning."""
    if generated_html is None: # Handle case where extraction failed
        gen_clean = ""
    else:
        gen_clean = generated_html.strip().replace("\n", "").replace(" ", "")

    if reference_html is None: # Handle potentially missing reference
        ref_clean = ""
    else:
        ref_clean = reference_html.strip().replace("\n", "").replace(" ", "")

    # Handle empty strings
    if not gen_clean and not ref_clean:
        return 1.0 # Both empty is perfect match
    if not gen_clean or not ref_clean:
        return 0.0 # One empty, one not is zero match

    max_len = max(len(gen_clean), len(ref_clean))
    if max_len == 0:
        return 1.0 # Should be caught above, but safety check

    distance = Levenshtein.distance(gen_clean, ref_clean)
    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity) # Ensure non-negative

# --- Load Data Samples ---
samples = load_test_samples(TEST_DATA_PATH, IMAGE_DIR, NUM_SAMPLES)
sample_options = {sample["id"]: sample for sample in samples} if samples else {}
sample_choices = list(sample_options.keys())

# --- Gradio Functions ---
def update_sample_display(selected_id):
    """Updates the image and prompt display."""
    if not selected_id or selected_id not in sample_options:
        return None, "Please select a valid sample.", "(Reference HTML will appear after generation)", "N/A"
    sample = sample_options[selected_id]
    try:
        img = Image.open(sample["image_path"]).convert('RGB')
        return img, sample["prompt"], "(Reference HTML will appear after generation)", "N/A"
    except Exception as e:
        print(f"Error loading image {sample['image_path']}: {e}")
        return None, f"Error loading image: {e}", "(Error loading image)", "N/A"

def process_sample_inference(selected_id):
    """Processes the selected sample for inference and calculates similarity."""
    if not selected_id or selected_id not in sample_options:
        return "No sample selected.", "(No HTML Preview)", "(No Reference HTML)", "N/A"

    sample = sample_options[selected_id]
    prompt = sample["prompt"]
    reference_response = sample.get("reference_response", "")

    try:
        image = Image.open(sample["image_path"]).convert('RGB')
    except Exception as e:
        error_msg = f"Could not load image: {e}"
        return error_msg, error_msg, "(Error loading image)", "N/A"

    print(f"Running inference for: {selected_id}")
    start_time = time.time()
    generated_text = run_vllm_inference(prompt, image)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference took {inference_time:.2f} seconds.")

    # Extract HTML from generated and reference text
    generated_html = extract_html(generated_text)
    reference_html = extract_html(reference_response)

    # Calculate similarity
    similarity_score = calculate_similarity(generated_html, reference_html)
    similarity_str = f"{similarity_score:.4f}"

    # Prepare display outputs
    html_preview = generated_html if generated_html else "(No valid HTML found in generation)"
    reference_preview = reference_html if reference_html else "(No valid HTML found in reference)"

    return generated_text, html_preview, reference_preview, similarity_str

def process_custom_inference(image_upload, prompt_input):
    """Processes custom input for inference."""
    if image_upload is None:
        return "Please upload an image.", "(No HTML Preview)"
    if not prompt_input:
         return "Please enter a prompt.", "(No HTML Preview)"

    image = image_upload
    prompt = prompt_input

    print(f"Running custom inference")
    start_time = time.time()
    generated_text = run_vllm_inference(prompt, image)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference took {inference_time:.2f} seconds.")

    html_content = extract_html(generated_text)
    html_preview = html_content if html_content else "(No valid HTML found)"
    
    return generated_text, html_preview

def chat_with_image(image_path, message):
    """Simple function for general chat with the model."""
    if not image_path:
        return "Please upload an image first."
    
    try:
        image = Image.open(image_path).convert('RGB')
        response = run_vllm_inference(message, image)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# --- Evaluation Function ---
def evaluate_all_samples():
    """Runs inference on all loaded samples and calculates average similarity."""
    if not samples:
        return "No samples loaded to evaluate."

    total_similarity = 0.0
    evaluated_count = 0
    error_count = 0
    results_summary = []
    all_scores = []
    rolling_averages = []

    print(f"Starting evaluation for {len(samples)} samples...")
    start_time_total = time.time()

    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        prompt = sample["prompt"]
        reference_response = sample.get("reference_response", "")
        print(f"Processing sample {i+1}/{len(samples)}: {sample_id}")

        try:
            image = Image.open(sample["image_path"]).convert('RGB')
            generated_text = run_vllm_inference(prompt, image)

            generated_html = extract_html(generated_text)
            reference_html = extract_html(reference_response)

            similarity_score = calculate_similarity(generated_html, reference_html)
            if similarity_score > 0.0:  # Exclude 0 scores (likely server errors)
                total_similarity += similarity_score
                evaluated_count += 1
                all_scores.append(similarity_score)
                current_rolling_avg = sum(all_scores) / len(all_scores)
                rolling_averages.append((len(all_scores), current_rolling_avg))
                results_summary.append(f"- {sample_id}: {similarity_score:.4f}")
                print(f"  Similarity: {similarity_score:.4f}, Rolling Avg: {current_rolling_avg:.4f}")
            else:
                print(f"  Skipping score of 0.0 (likely server error)")
                error_count += 1
                results_summary.append(f"- {sample_id}: Error (Score 0.0)")

        except FileNotFoundError:
            print(f"  Error: Image file not found for {sample_id} at {sample['image_path']}")
            error_count += 1
            results_summary.append(f"- {sample_id}: Error (Image not found)")
        except Exception as e:
            print(f"  Error processing sample {sample_id}: {e}")
            error_count += 1
            results_summary.append(f"- {sample_id}: Error ({type(e).__name__})")

    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    print(f"Evaluation finished in {total_time:.2f} seconds.")

    if evaluated_count > 0:
        average_similarity = total_similarity / evaluated_count
        rolling_avg_summary = ", ".join([f"{n}: {avg:.4f}" for n, avg in rolling_averages[::max(1, len(rolling_averages)//10)]])
        
        result_str = (
            f"Evaluation Complete:\n"
            f"- Samples Processed: {evaluated_count}\n"
            f"- Samples with Errors: {error_count}\n"
            f"- Average Levenshtein Similarity: {average_similarity:.4f}\n"
            f"- Total Time: {total_time:.2f}s\n\n"
            f"Rolling Average (sample count: similarity):\n{rolling_avg_summary}\n\n"
        )
    elif error_count > 0:
         result_str = (
            f"Evaluation Failed:\n"
            f"- Could not process any samples successfully.\n"
            f"- Errors encountered: {error_count}\n"
            f"- Total Time: {total_time:.2f}s"
         )
    else:
        result_str = "No samples were evaluated (list might be empty or all had errors)."

    print(result_str)
    return result_str

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 📊 Sophris Table VLM Inference (via vLLM)
    **Status:** Connecting to existing vLLM server.
    **Connection:** vLLM OpenAI API at `{VLLM_API_BASE_URL}`
    **Note:** Ensure the vLLM server is running the correct model (`{VLLM_EXPECTED_MODEL_NAME}` or fine-tuned version).
    """) # Updated Markdown

    with gr.Tabs():
        # Tab 1: Table Extraction from samples
        with gr.Tab("Table Extraction (Samples)"):
            with gr.Row():
                with gr.Column(scale=1):
                    sample_dropdown = gr.Dropdown(
                        choices=sample_choices,
                        label="Select Sample",
                        value=sample_choices[0] if sample_choices else None,
                        interactive=bool(sample_choices)
                    )
                    image_display = gr.Image(label="Input Image", type="pil", interactive=False)
                    prompt_display = gr.Textbox(label="Prompt", interactive=False, lines=3)
                    evaluate_all_button = gr.Button("📊 Evaluate All Samples", variant="secondary", interactive=bool(sample_choices))
                with gr.Column(scale=2):
                    generate_sample_button = gr.Button("✨ Generate Table HTML", variant="primary", interactive=bool(sample_choices))
                    similarity_display = gr.Textbox(label="Similarity Score (Levenshtein)", value="N/A", interactive=False)
                    raw_output_textbox = gr.Textbox(label="Raw Model Output", lines=8, interactive=False)
                    with gr.Row():
                        html_output_display = gr.HTML(label="Rendered Generated HTML")
                        reference_html_display = gr.HTML(label="Rendered Reference HTML")
                    evaluation_results_display = gr.Textbox(label="Evaluation Results", lines=10, interactive=False, placeholder="Click 'Evaluate All Samples' to see results...")

        # Tab 2: Custom Table Extraction
        with gr.Tab("Table Extraction (Custom)"):
            with gr.Row():
                with gr.Column(scale=1):
                    custom_image_upload = gr.Image(label="Upload Image", type="pil", sources=["upload"])
                    custom_prompt_input = gr.Textbox(
                        label="Enter Table Extraction Prompt", 
                        lines=3,
                        placeholder="Extract the table from this image..."
                    )
                with gr.Column(scale=2):
                    generate_custom_button = gr.Button("✨ Generate Table HTML", variant="primary")
                    custom_raw_output = gr.Textbox(label="Raw Model Output", lines=10, interactive=False)
                    custom_html_display = gr.HTML(label="Rendered HTML Table")

        # Tab 3: General Chat
        with gr.Tab("General Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    chat_image_input = gr.Image(label="Upload Image", type="filepath")
                    chat_input = gr.Textbox(
                        label="Your Message", 
                        lines=2,
                        placeholder="Ask anything about the image..."
                    )
                    chat_button = gr.Button("🚀 Send", variant="primary")
                with gr.Column(scale=2):
                    chat_output = gr.Textbox(label="Model Response", lines=15, interactive=False)

    # --- Event handlers ---
    # Sample tab
    if sample_choices:
        sample_dropdown.change(
            fn=update_sample_display,
            inputs=[sample_dropdown],
            outputs=[image_display, prompt_display, reference_html_display, similarity_display]
        )

        generate_sample_button.click(
            fn=process_sample_inference,
            inputs=[sample_dropdown],
            outputs=[raw_output_textbox, html_output_display, reference_html_display, similarity_display]
        )

        evaluate_all_button.click(
            fn=evaluate_all_samples,
            inputs=[],
            outputs=[evaluation_results_display]
        )

        # Initial load
        demo.load(
            fn=update_sample_display,
            inputs=[sample_dropdown],
            outputs=[image_display, prompt_display, reference_html_display, similarity_display]
        )

    # Custom extraction tab
    generate_custom_button.click(
        fn=process_custom_inference,
        inputs=[custom_image_upload, custom_prompt_input],
        outputs=[custom_raw_output, custom_html_display]
    )

    # Chat tab
    chat_button.click(
        fn=chat_with_image,
        inputs=[chat_image_input, chat_input],
        outputs=[chat_output]
    )

# --- Launch ---
if __name__ == "__main__":
    print("Launching Gradio Interface...")
    print(f"Connecting to vLLM server at: {VLLM_API_BASE_URL}") # Updated launch message
    # print(f"Model: {model_desc}") # Removed model description from launch message
    demo.launch(server_port=8011, share=True, server_name="0.0.0.0") 