import gradio as gr
from PIL import Image
import json
import os
import re
import traceback
import openai
import base64
from io import BytesIO
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models import register_model
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import glob
import time

# --- Configuration ---
# Models & paths
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TRAINING_OUTPUT_DIR = "output/sophris_table_prod1" # Your LoRA weights
HUB_MODEL_ID_FALLBACK = "ChunkrAI/sophris-table-VLM" # Fallback if no local checkpoint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE_DIR = "./hf_cache"

# vLLM Server details
VLLM_HOST = "localhost"
VLLM_PORT = 8001
VLLM_API_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"

# Data configuration
DATA_DIR_BASE = "data/sophris-datasheet-table-extraction-azure-distill-v1"
DATASET_BASENAME = os.path.basename(DATA_DIR_BASE)
IMAGE_DIR = os.path.join(DATA_DIR_BASE, DATASET_BASENAME)
TEST_DATA_PATH = "data/llava-format/test.json"
NUM_SAMPLES = 20
MAX_NEW_TOKENS = 2048

# --- Model Loading Logic (from test_hf_model.py) ---
def find_latest_checkpoint(output_dir):
    """Finds the latest checkpoint directory based on step number."""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None

    checkpoints = [c for c in checkpoints if os.path.isdir(c)]
    if not checkpoints:
        return None

    try:
        latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    except (ValueError, AttributeError, TypeError):
        print(f"Warning: Could not determine latest checkpoint. Sorting alphabetically.")
        latest_checkpoint = max(checkpoints)

    if os.path.isdir(latest_checkpoint) and os.path.exists(os.path.join(latest_checkpoint, "adapter_config.json")):
        print(f"Found latest valid checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    else:
        print(f"Warning: Latest checkpoint is not a valid LoRA checkpoint.")
        return None

# Determine which model to use (LoRA or HF Hub)
latest_checkpoint_path = find_latest_checkpoint(TRAINING_OUTPUT_DIR)

if latest_checkpoint_path:
    model_identifier = latest_checkpoint_path
    model_desc = f"Using LoRA: {os.path.basename(latest_checkpoint_path)} on {BASE_MODEL_ID}"
    print(f"Found LoRA checkpoint: {model_identifier}")
else:
    model_identifier = HUB_MODEL_ID_FALLBACK
    model_desc = f"Using HF Hub model: {HUB_MODEL_ID_FALLBACK}"
    print(f"No valid LoRA checkpoint found. Using Hub model: {model_identifier}")

print(f"Model to be used with vLLM: {model_desc}")

# --- Initialize OpenAI client ---
try:
    client = openai.OpenAI(
        base_url=VLLM_API_BASE_URL,
        api_key="dummy"
    )
except openai.APIConnectionError as e:
    print(f"FATAL: Could not connect to vLLM OpenAI server at {VLLM_API_BASE_URL}")
    print(f"Error: {e}")
    print("Please ensure the vLLM server is running with the correct model.")
    exit(1)
except Exception as e:
    print(f"FATAL: Error initializing OpenAI client: {e}")
    exit(1)

# --- Data Loading ---
def load_test_samples(data_path, image_base_dir, num_samples):
    """Loads samples from the test JSON file."""
    print(f"Loading test samples from {data_path}...")
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
    loaded_count = 0
    for i, item in enumerate(data):
        if loaded_count >= num_samples:
            break
        try:
            human_message = next((conv['value'] for conv in item['conversations'] if conv['from'] == 'human'), '')
            human_message = human_message.replace("<image>", "").strip()

            image_filename = item.get("image")
            if not image_filename or not isinstance(image_filename, str):
                print(f"Skipping sample {i+1}: Invalid or missing image filename.")
                continue

            image_path = os.path.join(image_base_dir, image_filename)

            if not os.path.exists(image_path):
                print(f"Skipping sample {i+1}: Image file not found at {image_path}")
                continue

            samples.append({
                "id": f"Sample {i+1} ({image_filename})",
                "prompt": human_message,
                "image_path": image_path,
                "image_filename": image_filename
            })
            loaded_count += 1
        except Exception as e:
            print(f"Skipping sample {i+1} due to error: {e}")

    if not samples:
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

        print(f"Sending request to vLLM API, max_tokens={max_new_tokens}")

        completion = client.chat.completions.create(
            model=model_identifier,  # Will be ignored by vLLM but good practice
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0
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

# --- HTML Parsing ---
def extract_html(text):
    """Extracts HTML content from model output."""
    if not text or text.startswith("Inference Error:") or text.startswith("Processing Error:"):
            return None

    text_stripped = text.strip()

    # Try standard regex first
    strict_match = re.search(r"```(?:html)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if strict_match:
        print("Info: Found well-formed ```html``` block.")
        return strict_match.group(1).strip()

    # Helper to check if content starts like HTML
    def looks_like_html(content):
        content_lower_stripped = content.strip().lower()
        return content_lower_stripped.startswith(("<table", "<div", "<html", "<tr", "<td", "<p", "<ul", "<ol"))

    # Heuristic 1: Missing closing backticks
    start_match = re.search(r"```(?:html)?\s*", text, re.IGNORECASE)
    if start_match:
        potential_html = text[start_match.end():].strip()
        if looks_like_html(potential_html):
             print("Info: Found opening ``` but no closing ```.")
             return potential_html

    # Heuristic 2: Missing opening backticks
    if text_stripped.endswith("```"):
         end_pos = text.rfind("```")
         if end_pos != -1:
             potential_html = text[:end_pos].strip()
             if looks_like_html(potential_html):
                 print("Info: Found closing ``` but no opening ```.")
                 return potential_html

    # Heuristic 3: No backticks at all
    if looks_like_html(text_stripped):
        print("Info: No ``` found, but text looks like HTML.")
        return text_stripped

    print("Info: Could not extract HTML content.")
    return None

# --- Load Data Samples ---
samples = load_test_samples(TEST_DATA_PATH, IMAGE_DIR, NUM_SAMPLES)
sample_options = {sample["id"]: sample for sample in samples} if samples else {}
sample_choices = list(sample_options.keys())

# --- Gradio Functions ---
def update_sample_display(selected_id):
    """Updates the image and prompt display."""
    if not selected_id or selected_id not in sample_options:
        return None, "Please select a valid sample."
    sample = sample_options[selected_id]
    try:
        img = Image.open(sample["image_path"]).convert('RGB')
        return img, sample["prompt"]
    except Exception as e:
        print(f"Error loading image {sample['image_path']}: {e}")
        return None, f"Error loading image: {e}"

def process_sample_inference(selected_id):
    """Processes the selected sample for inference."""
    if not selected_id or selected_id not in sample_options:
        return "No sample selected.", "(No HTML Preview)"

    sample = sample_options[selected_id]
    prompt = sample["prompt"]
    
    try:
        image = Image.open(sample["image_path"]).convert('RGB')
    except Exception as e:
        error_msg = f"Could not load image: {e}"
        return error_msg, error_msg

    print(f"Running inference for: {selected_id}")
    generated_text = run_vllm_inference(prompt, image)
    
    html_content = extract_html(generated_text)
    html_preview = html_content if html_content else "(No valid HTML found)"
    
    return generated_text, html_preview

def process_custom_inference(image_upload, prompt_input):
    """Processes custom input for inference."""
    if image_upload is None:
        return "Please upload an image.", "(No HTML Preview)"
    if not prompt_input:
         return "Please enter a prompt.", "(No HTML Preview)"

    image = image_upload
    prompt = prompt_input

    print(f"Running custom inference")
    generated_text = run_vllm_inference(prompt, image)
    
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

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 📊 Sophris Table VLM Inference (via vLLM)
    **Status:** Using model: {model_desc}
    **Connection:** vLLM API at {VLLM_API_BASE_URL}
    """)

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
                with gr.Column(scale=2):
                    generate_sample_button = gr.Button("✨ Generate Table HTML", variant="primary", interactive=bool(sample_choices))
                    raw_output_textbox = gr.Textbox(label="Raw Model Output", lines=10, interactive=False)
                    html_output_display = gr.HTML(label="Rendered HTML Table")

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
            outputs=[image_display, prompt_display]
        )

        generate_sample_button.click(
            fn=process_sample_inference,
            inputs=[sample_dropdown],
            outputs=[raw_output_textbox, html_output_display]
        )

        # Initial load
        demo.load(
            fn=update_sample_display,
            inputs=[sample_dropdown],
            outputs=[image_display, prompt_display]
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
    print(f"Model: {model_desc}")
    demo.launch(server_port=8011, share=True, server_name="0.0.0.0") 