import gradio as gr
import torch
# Import specific model class and PeftModel
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image
import json
import os
import re
import traceback
import glob # To find checkpoints

# --- Configuration ---
# BASE_MODEL_ID is needed when loading LoRA checkpoints
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct" # Make sure this matches your training base
# Specify the directory where training checkpoints are saved
TRAINING_OUTPUT_DIR = "output/sophris_table_prod1_merged" # Default from run_pipeline.sh
HUB_MODEL_ID_FALLBACK = "ChunkrAI/sophris-table-VLM" # Fallback if no local checkpoint found
DATA_DIR_BASE = "data/sophris-datasheet-table-extraction-azure-distill-v1"
DATASET_BASENAME = os.path.basename(DATA_DIR_BASE)
IMAGE_DIR = os.path.join(DATA_DIR_BASE, DATASET_BASENAME)
TEST_DATA_PATH = "data/llava-format/test.json"
NUM_SAMPLES = 10
MAX_NEW_TOKENS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CACHE_DIR = "./hf_cache"

# --- Helper Function to Find Latest Checkpoint (from evaluation logic) ---
def find_latest_checkpoint(output_dir):
    """Finds the latest checkpoint directory based on step number."""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None

    # Filter out any potential files, ensure we only consider directories
    checkpoints = [c for c in checkpoints if os.path.isdir(c)]
    if not checkpoints:
        return None

    try:
        # Sort by the integer step number
        latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'checkpoint-(\d+)', x).group(1)))
    except (ValueError, AttributeError, TypeError):
         print(f"Warning: Could not reliably determine latest checkpoint step number from {checkpoints}. Sorting alphabetically.")
         latest_checkpoint = max(checkpoints) # Fallback to alphabetical sort if numbering fails


    # Check if it's actually a directory and contains adapter config (sanity check for LoRA)
    if os.path.isdir(latest_checkpoint) and os.path.exists(os.path.join(latest_checkpoint, "adapter_config.json")):
        print(f"Found latest valid checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    else:
        print(f"Warning: Latest checkpoint candidate {latest_checkpoint} is not a valid LoRA checkpoint directory.")
        # Optionally, search backwards through sorted checkpoints if needed
        return None

# --- Updated Model Loading (Inspired by evaluation.py) ---
def load_model_and_processor(model_path_or_id, base_model_id_for_lora, use_flash_attn, cache_dir=None):
    """Loads the model and processor from a local checkpoint or Hub ID, mirroring evaluation.py logic."""
    print(f"Attempting to load model/processor using source: '{model_path_or_id}'")

    # Check if it's a LoRA model by looking for adapter_config.json in the path
    is_local_path = os.path.isdir(model_path_or_id)
    is_lora = is_local_path and os.path.exists(os.path.join(model_path_or_id, "adapter_config.json"))
    attn_implementation = "flash_attention_2" if use_flash_attn and torch.cuda.is_available() else "sdpa"

    try:
        if is_lora:
            print("Loading a LoRA model...")
            if not base_model_id_for_lora:
                 raise ValueError("base_model_id_for_lora must be provided when loading a LoRA checkpoint.")

            # Try to get base model from adapter config, but fallback to provided arg
            try:
                adapter_config = json.load(open(os.path.join(model_path_or_id, "adapter_config.json")))
                base_model_name = adapter_config.get("base_model_name_or_path", base_model_id_for_lora)
                if base_model_name != base_model_id_for_lora:
                    print(f"Note: Base model from adapter_config ({base_model_name}) differs from argument ({base_model_id_for_lora}). Using argument.")
                    base_model_name = base_model_id_for_lora # Prioritize explicit argument
            except Exception as e:
                 print(f"Warning: Could not read base model from adapter_config.json: {e}. Using provided base: {base_model_id_for_lora}")
                 base_model_name = base_model_id_for_lora

            print(f"Using base model: {base_model_name}")
            # Load processor from base model
            processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=cache_dir)

            # Load base model (Using specific class like in evaluation.py)
            # Assuming Qwen2.5-VL, adjust if using original Qwen-VL
            print(f"Loading base model weights for {base_model_name}...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=DEVICE, # Use device map for potential multi-GPU or offloading
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                cache_dir=cache_dir
            )
            print(f"Loading LoRA adapter from {model_path_or_id}...")
            # Apply the LoRA adapter
            model = PeftModel.from_pretrained(model, model_path_or_id)
            print("LoRA adapter loaded successfully!")

        else:
            # Assume it's a Hub ID or a fully merged local model path
            print(f"Loading a regular (non-LoRA) model from: {model_path_or_id}")
            # Load processor FROM THE BASE MODEL ID if available, otherwise fallback to model_path_or_id
            # This helps when the target model's config.json is missing 'model_type'
            processor_source_id = BASE_MODEL_ID if BASE_MODEL_ID else model_path_or_id
            print(f"Loading processor using source: {processor_source_id}")
            processor = AutoProcessor.from_pretrained(processor_source_id, trust_remote_code=True, cache_dir=cache_dir)

            # Load the full model (Use specific class for consistency)
            print(f"Loading full model weights for {model_path_or_id}...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path_or_id,
                torch_dtype=torch.bfloat16,
                device_map=DEVICE,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                cache_dir=cache_dir
            )
            print("Full model loaded successfully.")

        model.eval()
        return model, processor

    except Exception as e:
        print(f"Error loading model: {e}\n{traceback.format_exc()}")
        raise # Re-raise exception

# --- Data Loading (Unchanged) ---
# ... existing load_test_samples function ...
def load_test_samples(data_path, image_base_dir, num_samples):
    """Loads the first N samples from the test JSON file."""
    print(f"Loading test samples from {data_path}...")
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {data_path}")
        print("Please ensure the dataset is prepared and the path is correct.")
        return [] # Return empty list on error
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_path}")
        return [] # Return empty list on error

    samples = []
    for i, item in enumerate(data):
        if len(samples) >= num_samples: # Stop once we have enough valid samples
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
                "id": f"Sample {i+1} ({image_filename})", # Use this as the display name/key
                "prompt": human_message,
                "image_path": image_path,
                "image_filename": image_filename
            })
        except Exception as e:
            print(f"Skipping sample {i+1} due to error: {e}")

    if not samples:
        print("Warning: No valid samples could be loaded. Check data format and image paths.")

    print(f"Loaded {len(samples)} samples.")
    return samples


# --- Inference (Unchanged - suitable for interactive app) ---
# ... existing run_inference function ...
def run_inference(model, processor, prompt: str, image: Image.Image, max_new_tokens=1024):
    """Runs inference on a single image and prompt."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            input_token_len = inputs['input_ids'].shape[1]
            generated_ids_trimmed = generated_ids[:, input_token_len:]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        return response.strip()
    except Exception as e:
        print(f"Inference error: {e}\n{traceback.format_exc()}")
        return f"Inference Error: {e}" # Return error message string


# --- HTML Parsing (Updated with Heuristics) ---
def extract_html(text):
    """
    Extracts HTML content from ```html ... ``` blocks.
    Includes heuristics to handle missing opening or closing backticks
    if the content appears to be HTML.
    """
    if not text or text.startswith("Inference Error:") or text.startswith("Processing Error:"):
            return None # Don't try to parse error messages

    text_stripped = text.strip() # Work with the stripped version for some checks

    # 1. Try the standard regex first (most reliable for well-formed blocks)
    strict_match = re.search(r"```(?:html)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if strict_match:
        print("Info: Found well-formed ```html``` block.")
        return strict_match.group(1).strip()

    # Helper to check if content starts like common HTML tags
    def looks_like_html(content):
        content_lower_stripped = content.strip().lower()
        # Check for common block/table elements
        return content_lower_stripped.startswith(("<table", "<div", "<html", "<tr", "<td", "<p", "<ul", "<ol"))

    # 2. Heuristic: Missing closing backticks ```
    # Find the start marker ` ``` ` or ` ```html `
    start_match = re.search(r"```(?:html)?\s*", text, re.IGNORECASE)
    if start_match:
        # Extract content *after* the opening marker
        potential_html = text[start_match.end():].strip()
        # Check if this extracted content looks like HTML
        if looks_like_html(potential_html):
             print("Info: Found opening ``` but no closing ```. Attempting heuristic extraction.")
             return potential_html # Return the stripped content after the opening ticks

    # 3. Heuristic: Missing opening backticks ``` (Check if it ends with ```)
    # Look for closing backticks potentially at the end
    if text_stripped.endswith("```"):
         # Find the position of the *last* occurrence of ```
         # Use rfind on the original text to get the correct index before stripping might affect it
         end_pos = text.rfind("```")
         if end_pos != -1:
             # Extract content *before* the closing marker
             potential_html = text[:end_pos].strip()
             # Check if this extracted content looks like HTML
             if looks_like_html(potential_html):
                 print("Info: Found closing ``` but no opening ```. Attempting heuristic extraction.")
                 return potential_html # Return the stripped content before the closing ticks

    # 4. Heuristic: No backticks at all, but the whole text starts like HTML
    if looks_like_html(text_stripped):
        print("Info: No ``` found, but text starts like HTML. Returning raw stripped text.")
        return text_stripped # Assume the whole stripped text is the HTML

    # 5. Fallback: Nothing reasonably identifiable as HTML found
    print("Info: Could not reliably extract HTML content using standard or heuristic methods.")
    return None


# --- Load Model and Data ---
# Find the latest checkpoint # << COMMENT OUT OR REMOVE
latest_checkpoint_path = find_latest_checkpoint(TRAINING_OUTPUT_DIR) # << COMMENT OUT OR REMOVE

# Decide which model identifier to use
model_identifier = None # << COMMENT OUT OR REMOVE
base_for_loader = None # << COMMENT OUT OR REMOVE
status_message = "" # << COMMENT OUT OR REMOVE
use_flash_attn = True # Default from evaluation.py args

# --- FORCE HUB MODEL ---
# Always use the specified Hub model ID, ignore local checkpoints
model_identifier = HUB_MODEL_ID_FALLBACK
base_for_loader = None # Not needed when loading directly from Hub/full model
status_message = f"Forcing load from Hub: `{model_identifier}`"
print(status_message)
# --- END FORCE HUB MODEL ---

# << REMOVE OR COMMENT OUT THE if/else BLOCK BELOW >>
if latest_checkpoint_path:
    model_identifier = latest_checkpoint_path
    base_for_loader = BASE_MODEL_ID # MUST be provided for LoRA
    status_message = f"Loading local checkpoint: `{os.path.basename(model_identifier)}` (Base: `{base_for_loader}`)"
    print(status_message)
else:
    print(f"No valid checkpoint found in {TRAINING_OUTPUT_DIR}. Falling back to Hub model: {HUB_MODEL_ID_FALLBACK}")
    model_identifier = HUB_MODEL_ID_FALLBACK
    base_for_loader = None # Not needed when loading directly from Hub/full model
    status_message = f"Loading from Hub: `{model_identifier}` (No local checkpoint found)"
    print(status_message)
# << END OF REMOVED/COMMENTED OUT BLOCK >>


# Load the model using the chosen identifier and logic
try:
    model, processor = load_model_and_processor(
        model_path_or_id=model_identifier,
        base_model_id_for_lora=base_for_loader, # Pass base model explicitly (will be None here)
        use_flash_attn=use_flash_attn,
        cache_dir=MODEL_CACHE_DIR
    )
except Exception as e:
    print(f"FATAL: Failed to load model. Exiting. Error: {e}")
    # Provide more specific error message if it's a LoRA loading issue without base model
    # if latest_checkpoint_path and not base_for_loader: # << CAN REMOVE THIS CHECK NOW >>
    #      print("Hint: Ensure BASE_MODEL_ID is correctly set in the script when loading a LoRA checkpoint.") # << CAN REMOVE THIS HINT NOW >>
    exit(1) # Exit if model loading fails critically

# Load data samples
samples = load_test_samples(TEST_DATA_PATH, IMAGE_DIR, NUM_SAMPLES)
if not samples:
    print("FATAL: No data samples loaded. Exiting.")
    exit(1) # Exit if data loading fails

# Create a mapping from sample ID to sample data for easy lookup
sample_options = {sample["id"]: sample for sample in samples}
sample_choices = list(sample_options.keys())

# --- Gradio Specific Functions (Unchanged) ---
# ... existing update_sample_display function ...
def update_sample_display(selected_id):
    """Updates the image and prompt display when a sample is selected."""
    if not selected_id or selected_id not in sample_options:
        return None, "Please select a valid sample."
    sample = sample_options[selected_id]
    try:
        img = Image.open(sample["image_path"]).convert('RGB')
        return img, sample["prompt"]
    except Exception as e:
        print(f"Error loading image {sample['image_path']}: {e}")
        return None, f"Error loading image: {e}"

# ... existing process_inference function ...
def process_inference(selected_id):
    """Handles the full inference process for the selected sample."""
    if not selected_id or selected_id not in sample_options:
        return "No sample selected or invalid ID.", "Please select a sample first."

    sample = sample_options[selected_id]
    prompt = sample["prompt"]
    image_path = sample["image_path"]

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image for inference {image_path}: {e}")
        return f"Processing Error: Could not load image {sample['image_filename']}", f"Error: {e}"

    print(f"Running inference for: {selected_id}")
    generated_text = run_inference(model, processor, prompt, image, MAX_NEW_TOKENS)
    print("Inference complete.")

    print("Extracting HTML...")
    html_content = extract_html(generated_text)
    print(f"HTML extracted: {'Yes' if html_content else 'No'}")

    if html_content:
        # Return raw text and the HTML content for rendering
            return generated_text, html_content
    elif generated_text.startswith("Inference Error:") or generated_text.startswith("Processing Error:"):
            # Pass through errors
            return generated_text, generated_text
    else:
        # Return raw text and a message indicating no HTML found
        return generated_text, "(No valid HTML table found in the output)"

def chat_with_image(model, processor, image_path: str, message: str, max_new_tokens=1024):
    """Simple function to chat with the model about an image."""
    try:
        image = Image.open(image_path).convert('RGB')
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": message}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            input_token_len = inputs['input_ids'].shape[1]
            generated_ids_trimmed = generated_ids[:, input_token_len:]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        return response.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 📊 Sophris Table VLM\nStatus: {status_message}")

    with gr.Tabs():  # Use Tabs to organize different functionalities
        # Tab 1: Table Extraction
        with gr.Tab("Table Extraction"):
            with gr.Row():
                with gr.Column(scale=1):
                    sample_dropdown = gr.Dropdown(
                        choices=sample_choices,
                        label="Select Sample",
                        value=sample_choices[0] if sample_choices else None
                    )
                    image_display = gr.Image(label="Input Image", type="pil")
                    prompt_display = gr.Textbox(label="Prompt", interactive=False, lines=3)

                with gr.Column(scale=2):
                    generate_button = gr.Button("✨ Generate Table HTML", variant="primary")
                    raw_output_textbox = gr.Textbox(label="Raw Output", lines=10)
                    html_output_display = gr.HTML(label="Rendered Table")

        # Tab 2: General Chat
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    chat_image_input = gr.Image(label="Upload Image", type="filepath")
                    chat_input = gr.Textbox(
                        label="Your Message",
                        lines=2,
                        placeholder="Ask anything about the image..."
                    )
                    chat_button = gr.Button("🚀 Send Message", variant="primary")
                
                with gr.Column(scale=2):
                    chat_history = gr.Textbox(
                        label="Chat History",
                        lines=15,
                        interactive=False
                    )

    # Event handlers
    sample_dropdown.change(
        fn=update_sample_display,
        inputs=[sample_dropdown],
        outputs=[image_display, prompt_display]
    )

    generate_button.click(
        fn=process_inference,
        inputs=[sample_dropdown],
        outputs=[raw_output_textbox, html_output_display]
    )

    # Add chat handler
    chat_button.click(
        fn=lambda img, msg: chat_with_image(model, processor, img, msg),
        inputs=[chat_image_input, chat_input],
        outputs=[chat_history]
    )

    # Initial load for table extraction
    if sample_choices:
        demo.load(
            fn=update_sample_display,
            inputs=[sample_dropdown],
            outputs=[image_display, prompt_display]
        )

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio Interface...")
    # You can specify server_name="0.0.0.0" to make it accessible on your network
    # Set share=False unless you need a public link (requires login/organization)
    demo.launch(server_port=8011, share=True, server_name="0.0.0.0")