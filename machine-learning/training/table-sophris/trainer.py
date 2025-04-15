import os
import torch
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, TrainerCallback, AutoTokenizer
from dotenv import load_dotenv
import logging
from datasets import Dataset, Features, Value, Sequence, Image as HFImage, load_dataset
from PIL import Image, UnidentifiedImageError
import io
import random
import base64
import time
from tqdm.auto import tqdm
from datetime import datetime
import sys
import importlib.util
from huggingface_hub import model_info
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced callback for progress display and stats collection
class StatsTrackingCallback(TrainerCallback):
    def __init__(self, run_name=None):
        self.training_bar = None
        self.last_log = {}
        self.start_time = time.time()
        
        # Initialize stats tracking
        self.step_history = []
        self.loss_history = []
        self.lr_history = []
        self.time_history = []
        
        # Create run name with timestamp
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"run_{timestamp}"
        else:
            self.run_name = run_name
            
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.training_bar = tqdm(total=args.max_steps, desc="Training")
        logger.info(f"Training started - Run: {self.run_name}")
        
        # Save initial configuration
        self.args = args
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called on each log event"""
        if logs:
            # Extract loss from logs - handle both dictionary and tensor cases
            if 'loss' in logs:
                current_loss = logs['loss']
                # Convert from tensor if needed
                if hasattr(current_loss, 'item'):
                    current_loss = current_loss.item()
                self.loss_history.append(float(current_loss))
            else:
                self.loss_history.append(float('nan'))
            
            # Track learning rate
            if 'learning_rate' in logs:
                self.lr_history.append(logs['learning_rate'])
            else:
                self.lr_history.append(float('nan'))
            
            # Track step and time
            self.step_history.append(state.global_step)
            self.time_history.append(time.time() - self.start_time)
            
            # Update progress bar description
            if self.training_bar is not None:
                desc = f"Step: {state.global_step}"
                if 'loss' in logs:
                    desc += f" | Loss: {current_loss:.4f}"
                if 'learning_rate' in logs:
                    desc += f" | LR: {logs['learning_rate']:.2e}"
                self.training_bar.set_description(desc)
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.training_bar:
            self.training_bar.close()
            total_time = time.time() - self.start_time
            logger.info(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
            
            # Save run statistics
            self._save_run_stats(args, state, total_time)
            
    def _save_run_stats(self, args, state, total_time):
        """Save all training run statistics and visualizations"""
        # Create training_runs directory if it doesn't exist
        runs_dir = "training_runs"
        os.makedirs(runs_dir, exist_ok=True)
        
        # Create a directory for this run
        run_dir = os.path.join(runs_dir, self.run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # 1. Save configuration as JSON
        config_dict = vars(args)
        # Convert non-serializable objects to strings
        for key, value in config_dict.items():
            if not isinstance(value, (int, float, str, bool, type(None), list, dict)):
                config_dict[key] = str(value)
                
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        # 2. Save step metrics as CSV
        metrics_df = pd.DataFrame({
            'step': self.step_history,
            'loss': self.loss_history,
            'learning_rate': self.lr_history,
            'time_elapsed': self.time_history
        })
        metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
        
        # 3. Create and save loss/learning rate curve
        if len(self.step_history) > 0:
            plt.figure(figsize=(12, 8))
            
            # Loss subplot
            plt.subplot(2, 1, 1)
            plt.plot(self.step_history, self.loss_history, 'b-', label='Training Loss')
            plt.title(f'Training Loss Curve - {self.run_name}')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Learning rate subplot
            plt.subplot(2, 1, 2)
            plt.plot(self.step_history, self.lr_history, 'g-', label='Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Steps')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "training_curves.png"))
            plt.close()
            
        # 4. Save summary statistics
        summary = {
            "run_name": self.run_name,
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(self.start_time + total_time).strftime("%Y-%m-%d %H:%M:%S"),
            "total_training_time_seconds": total_time,
            "total_steps_completed": state.global_step,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "model_name": args.model_name_or_path if hasattr(args, "model_name_or_path") else "unknown",
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
        }
        
        with open(os.path.join(run_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Training run statistics saved to {run_dir}")

# Define a custom dataset class that loads images on-the-fly
class CustomChatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.min_dim = 32  # Minimum dimension requirement (increased from 28 for safety)
        self.divisible_by = 32

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        messages = []
        
        for message in example['messages']:
            content = []
            for item in message['content']:
                if item.get('type') == 'image_base64':
                    img_base64 = item.get('image_base64')
                    if img_base64:
                        try:
                            img_bytes = base64.b64decode(img_base64)
                            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            
                            # Get original dimensions
                            width, height = image.size
                            
                            # Calculate scaling factor to ensure minimum dimensions
                            width_scale = self.min_dim / width if width < self.min_dim else 1
                            height_scale = self.min_dim / height if height < self.min_dim else 1
                            scale = max(width_scale, height_scale)
                            
                            # Calculate new dimensions
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            
                            # Make dimensions divisible by divisible_by
                            new_width = ((new_width + self.divisible_by - 1) // self.divisible_by) * self.divisible_by
                            new_height = ((new_height + self.divisible_by - 1) // self.divisible_by) * self.divisible_by
                            
                            # Resize image
                            image = image.resize((new_width, new_height), Image.LANCZOS)
                            
                            content.append({"type": "image", "image": image, "text": None})
                        except Exception as e:
                            logger.warning(f"Failed to load/resize image in __getitem__: {e}")
                else:
                    content.append(item)
            
            messages.append({"role": message['role'], "content": content})
        
        return {"messages": messages}

# Add this function after imports
def ensure_model_downloaded(model_name, cache_dir=None):
    """
    Ensures the model is downloaded before attempting to load it.
    
    Args:
        model_name: The name of the model on HuggingFace or a local path
        cache_dir: Optional custom cache directory
    
    Returns:
        True if model is available, False otherwise
    """
    try:
        # For HuggingFace models, try to download the tokenizer which is smaller
        if '/' in model_name and not os.path.exists(model_name):
            logger.info(f"Checking if model {model_name} is available...")
            
            from transformers import AutoTokenizer
            from huggingface_hub import model_info
            
            # Check if model exists on hub
            try:
                info = model_info(model_name)
                logger.info(f"Model found on HuggingFace: {info.modelId}")
                
                # Download tokenizer to ensure model is accessible
                logger.info(f"Downloading tokenizer for {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                logger.info(f"Tokenizer downloaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to check model availability: {e}")
                return False
        elif os.path.exists(model_name):
            logger.info(f"Model exists at local path: {model_name}")
            return True
        else:
            # For local paths that don't exist
            logger.error(f"Model not found at: {model_name}")
            return False
    except Exception as e:
        logger.error(f"Error checking model: {e}")
        return False

# Replace the dataset loading section with this implementation
def load_jsonl_directly(file_path):
    """
    Load JSONL file directly, expanding path patterns if needed.
    """
    import glob
    import json
    
    # Handle [0-9] pattern in file path
    if '[0-9]' in file_path:
        # Expand the pattern to find the actual file
        matching_files = glob.glob(file_path.replace('[0-9]', '?'))
        if matching_files:
            file_path = matching_files[0]
            logger.info(f"Expanded path pattern to: {file_path}")
        else:
            logger.error(f"No files match pattern: {file_path}")
            return []
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []
    
    # Load records
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} records from {file_path}")
        return records
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

# --- Main Training Function ---
def main():
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(description="Train table OCR models")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-VL-3B-Instruct", 
                        help="Base model to finetune")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2_5_Vl_3B-table-finetune", 
                        help="Directory for checkpoints")
    parser.add_argument("--lora_output_dir", type=str, default="lora_model/qwen2_5_Vl_3B-table-finetune", 
                        help="Directory for final LoRA model")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this training run (defaults to timestamp)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="all-linear", help="LoRA target modules ('all-linear' or comma-separated)")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=25, help="Save checkpoint every N steps")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory containing train/val/test split data")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of the dataset subdirectory within data_dir")
    parser.add_argument("--prepared_data_file", type=str, default=None, 
                        help="Path to legacy JSONL data file (if not using train/val/test split)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    # Clean up any quotes in string arguments
    if args.data_dir and args.data_dir.startswith('"') and args.data_dir.endswith('"'):
        args.data_dir = args.data_dir[1:-1]
    
    if args.dataset_name and args.dataset_name.startswith('"') and args.dataset_name.endswith('"'):
        args.dataset_name = args.dataset_name[1:-1]
        
    # Get dataset path
    data_dir = args.data_dir if args.data_dir else "data"
    dataset_name = args.dataset_name if args.dataset_name else "default"
    
    logger.info(f"Using data_dir: {data_dir}")
    logger.info(f"Using dataset_name: {dataset_name}")
    
    # Build exact dataset path - keep brackets but strip any quotes
    dataset_dir = os.path.join(data_dir, dataset_name)
    train_file = os.path.join(dataset_dir, "train.jsonl")
    
    logger.info(f"Looking for dataset at: {dataset_dir}")
    logger.info(f"Looking for training file at: {train_file}")
    
    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        
        # List available datasets to help debugging
        try:
            if os.path.exists(data_dir):
                available_datasets = [d for d in os.listdir(data_dir) 
                                     if os.path.isdir(os.path.join(data_dir, d))]
                logger.error(f"Available datasets in {data_dir}: {available_datasets}")
            else:
                logger.error(f"Data directory does not exist: {data_dir}")
                logger.error(f"Current working directory: {os.getcwd()}")
        except Exception as e:
            logger.error(f"Error listing directories: {e}")
        
        # Try to use default dataset as fallback
        default_dir = os.path.join(data_dir, "default")
        if os.path.exists(default_dir) and os.path.isfile(os.path.join(default_dir, "train.jsonl")):
            logger.info(f"Using default dataset as fallback: {default_dir}")
            dataset_dir = default_dir
            train_file = os.path.join(dataset_dir, "train.jsonl")
        else:
            logger.error("No fallback dataset found, exiting")
            return

    # Create standardized output directory path
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    
    # Create standardized lora output path
    model_name_safe = args.model_name.replace("/", "_").replace("\\", "_")
    lora_output_dir = args.lora_output_dir or os.path.join(output_dir, f"lora_{model_name_safe}")
    logger.info(f"LoRA will be saved to: {lora_output_dir}")
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(lora_output_dir, exist_ok=True)
    
    # --- Configuration ---
    model_name = args.model_name
    run_name = args.run_name
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    max_steps = args.max_steps
    learning_rate = args.lr
    max_seq_length = args.max_seq_length
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_target_modules_str = args.lora_target_modules
    logging_steps = args.logging_steps
    save_steps = args.save_steps
    warmup_steps = args.warmup_steps
    seed = args.seed
    data_dir = args.data_dir
    dataset_name = args.dataset_name or os.environ.get("DATASET_NAME", "default")
    prepared_data_file = args.prepared_data_file

    # Make sure model is accessible
    cache_dir = os.environ.get("TRANSFORMERS_CACHE", "model_cache")
    model_exists = ensure_model_downloaded(model_name, cache_dir)
    
    if not model_exists:
        logger.error(f"Model {model_name} could not be accessed. Please check the model name or internet connection.")
        return
        
    # --- Load Model and Tokenizer ---
    logger.info(f"Loading base model: {model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None, # Auto-detect
        load_in_4bit = True,
    )
    logger.info("Base model loaded.")

    # --- Add LoRA ---
    logger.info("Adding LoRA adapters...")
    lora_target_modules = None
    if lora_target_modules_str and lora_target_modules_str.lower() != "all-linear":
        lora_target_modules = [m.strip() for m in lora_target_modules_str.split(',')]

    model = FastVisionModel.get_peft_model(
        model,
        r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        target_modules = lora_target_modules if lora_target_modules else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "mlp.fc1", "mlp.fc2"], # Default Qwen VL targets
        lora_target_all_linear_layers = (lora_target_modules is None), # True if default or 'all-linear'
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = seed,
        max_seq_length = max_seq_length,
    )
    logger.info("LoRA adapters added.")
    model.print_trainable_parameters()

    # --- Load Pre-processed Dataset ---
    logger.info("Loading pre-processed dataset...")

    # Load dataset directly from train.jsonl
    try:
        # Read records directly from JSONL
        with open(train_file, 'r', encoding='utf-8') as f:
            raw_records = [json.loads(line) for line in f if line.strip()]
        
        logger.info(f"Loaded {len(raw_records)} records from {train_file}")
        
        # Create dataset directly
        from datasets import Dataset
        
        # Extract messages from records
        messages = []
        table_ids = []
        
        for i, record in enumerate(raw_records):
            if "messages" in record:
                messages.append(record["messages"])
                table_ids.append(record.get("table_id", f"id_{i}"))
        
        # Create dataset
        raw_dataset = Dataset.from_dict({
            "messages": messages,
            "table_id": table_ids
        })
        
        logger.info(f"Created dataset with {len(raw_dataset)} samples")
        
        # Create our custom dataset
        final_dataset = CustomChatDataset(raw_dataset)
        logger.info(f"Final dataset created with {len(final_dataset)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # --- Configure Trainer with Stats Callback ---
    logger.info("Configuring SFTTrainer...")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create stats tracking callback
    stats_callback = StatsTrackingCallback(run_name=run_name)

    # Use the standard Unsloth collator with our custom dataset
    data_collator = UnslothVisionDataCollator(model, tokenizer)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        report_to="none",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        dataset_text_field="messages",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=max_seq_length,
        dataset_num_proc = 4,

    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=final_dataset, # Use the custom dataset that loads images on-the-fly
        data_collator=data_collator,
        args=training_args,
        callbacks=[stats_callback],  # Add our stats callback
    )
    logger.info("SFTTrainer configured.")

    # --- Train ---
    logger.info("Starting training...")
    FastVisionModel.for_training(model) # Prepare model for training
    trainer_stats = trainer.train()
    logger.info("Training finished.")
    logger.info(f"Trainer stats: {trainer_stats}")

    # --- Save Model ---
    logger.info(f"Saving LoRA adapters to {lora_output_dir}")
    os.makedirs(lora_output_dir, exist_ok=True)
    
    # Ensure model is in the correct state before saving
    try:
        logger.info("Preparing model for saving...")        
        # Save the model
        logger.info("Saving model...")
        model.save_pretrained(lora_output_dir)
        tokenizer.save_pretrained(lora_output_dir)
        
        # Save training arguments for future reference
        with open(os.path.join(lora_output_dir, "training_args.json"), "w") as f:
            json.dump({
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "max_steps": max_steps,
                "dataset": dataset_name,
                "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
            
        logger.info("Model saved successfully.")
        
        # Verify that the model files exist
        if os.path.exists(os.path.join(lora_output_dir, "adapter_model.bin")):
            logger.info("Verified: adapter_model.bin exists")
        else:
            logger.warning("Warning: adapter_model.bin not found after saving!")
            
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        logger.error("Model saving failed. Continuing with evaluation if requested.")

    # --- Run Evaluation ---
    logger.info("Starting evaluation of fine-tuned model...")
    
    # Dynamically import the eval module
    try:
        # Get the absolute path to eval.py in the same directory as this script
        eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
        
        # Set up the module spec and load the module
        spec = importlib.util.spec_from_file_location("eval_module", eval_path)
        eval_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_module)
        
        # Get the number of evaluation samples (use a reasonable default)
        num_eval_samples = 10
        
        # Create an evaluation output folder within the training run directory
        eval_output_folder = os.path.join("training_runs", stats_callback.run_name, "evaluation")
        os.makedirs(eval_output_folder, exist_ok=True)
        
        # Run the evaluation on the TEST set (not validation set)
        eval_result = eval_module.run_evaluation(
            model_name_or_path=lora_output_dir,
            num_eval_samples=num_eval_samples,
            base_model_name=model_name,
            batch_size=batch_size,
            output_folder=eval_output_folder,
            data_dir=data_dir,
            dataset_name=dataset_name,
            eval_split="test",
            compare_with_baseline=True,
            baseline_model=model_name
        )
        
        # Save evaluation summary to the training run directory
        if eval_result:
            if "baseline_similarity" in eval_result:
                logger.info(f"Evaluation complete. Model: {eval_result['average_similarity']:.4f}, Baseline: {eval_result['baseline_similarity']:.4f}, Improvement: {eval_result['improvement']:.4f}")
                
                # Save a link to the evaluation results in the training directory
                with open(os.path.join("training_runs", stats_callback.run_name, "eval_summary.txt"), "w") as f:
                    f.write(f"Evaluation results: {eval_output_folder}\n")
                    f.write(f"Model similarity: {eval_result['average_similarity']:.4f}\n")
                    f.write(f"Baseline similarity: {eval_result['baseline_similarity']:.4f}\n")
                    f.write(f"Improvement: {eval_result['improvement']:.4f} ({eval_result['improvement']*100:.1f}%)\n")
                    f.write(f"Number of samples: {eval_result['num_samples']}\n")
                    f.write(f"Exact matches: {sum(1 for r in eval_result['results'] if r['exact_match'])}/{len(eval_result['results'])}\n")
            else:
                # Regular summary without baseline comparison
                logger.info(f"Evaluation complete. Average similarity: {eval_result['average_similarity']:.4f}")
                # Save summary as in the original code
                with open(os.path.join("training_runs", stats_callback.run_name, "eval_summary.txt"), "w") as f:
                    f.write(f"Evaluation results: {eval_output_folder}\n")
                    f.write(f"Average similarity: {eval_result['average_similarity']:.4f}\n")
                    f.write(f"Number of samples: {eval_result['num_samples']}\n")
                    f.write(f"Exact matches: {sum(1 for r in eval_result['results'] if r['exact_match'])}/{len(eval_result['results'])}\n")
    
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        logger.error("Skipping evaluation. You can run it manually using eval.py")

if __name__ == "__main__":
    main()
