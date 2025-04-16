import os
import torch
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
from datasets import Dataset
from PIL import Image
import io
import random
import base64
import time
from tqdm.auto import tqdm
from datetime import datetime
import sys
import tempfile
import yaml
import shutil
import subprocess
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from data_loader import TableDatasetLoader

from accelerate import Accelerator, DeepSpeedPlugin
from torch.utils.data import DataLoader
from torch.optim import AdamW
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DeepSpeed Configuration ---
def setup_deepspeed():
    logger.info("Initializing DeepSpeed plugin...")
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=3,
        gradient_accumulation_steps=2,
        zero3_save_16bit_model=True,
        offload_optimizer_device="cpu",
        offload_param_device="cpu"
    )
    logger.info("Initializing Accelerator...")
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    logger.info(f"Device: {accelerator.device}")
    return accelerator

# Enhanced callback for progress display and stats collection
class StatsTrackingCallback:
    def __init__(self, run_name=None):
        self.loss_history = []
        self.step_history = []
        self.lr_history = []
        self.time_history = []
        self.start_time = time.time()
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def start_training(self, total_steps):
        """Record training start"""
        self.start_time = time.time()
        logger.info(f"Starting training run: {self.run_name} with {total_steps} steps")
        
    def log_step(self, step, loss, lr):
        """Record stats for a training step"""
        self.step_history.append(step)
        self.loss_history.append(loss)
        self.lr_history.append(lr)
        self.time_history.append(time.time() - self.start_time)
        
    def end_training(self, args, total_steps):
        """Handle end of training and save stats"""
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Saving run statistics...")
        self._save_run_stats(args, total_steps, total_time)
    
    def _save_run_stats(self, args, total_steps, total_time):
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
            "total_steps_completed": total_steps,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
        }
        
        with open(os.path.join(run_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Training run statistics saved to {run_dir}")

def find_assistant_content_sublist_indexes(l):
    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2)
                    break

    return list(zip(start_indexes, end_indexes))

def collate_fn(batch, processor, device):
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    
    # Process images from messages
    image_inputs = []
    for msg_list in messages:
        for msg in msg_list:
            if msg['role'] == 'user':
                for content_item in msg['content']:
                    if content_item.get('type') == 'image':
                        if isinstance(content_item['image'], Image.Image):
                            image_inputs.append(content_item['image'])
    
    inputs = processor(
        text=texts,
        images=image_inputs if image_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    
    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64).to(device)
    return inputs, labels_ids

def write_chat_template(processor, output_dir):
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"Chat template saved in {output_chat_template_file}")

# --- Argument Parsing ---
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Vision-Language Model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing the dataset folders.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset folder within data_dir.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of training samples to load (for debugging).")
    parser.add_argument("--eval_split", type=str, default="test", help="Which split to use for evaluation ('val' or 'test').")
    parser.add_argument("--num_eval_samples", type=int, default=200, help="Number of samples to use for evaluation.")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Hugging Face model identifier.")
    parser.add_argument("--local_model_path", type=str, default=None, help="Path to the locally downloaded base model directory.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save checkpoints and logs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps.")
    parser.add_argument("--max_steps", type=int, default=100, help="Total number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log training info every N steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # GPU Configuration
    parser.add_argument("--cuda_devices", type=str, default=None, help="Comma-separated list of CUDA device IDs to use (e.g., '0,1').")

    args = parser.parse_args()
    return args

# --- Main Training Function ---
def main():
    args = parse_arguments()
    load_dotenv()

    # --- Setup Directories ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Main output directory: {output_dir}")

    # --- DeepSpeed and Accelerator Setup ---
    accelerator = setup_deepspeed()

    # --- Load Dataset ---
    logger.info("===== LOADING DATASET =====")
    dataset_path = os.path.join(args.data_dir, args.dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # Use TableDatasetLoader
    loader = TableDatasetLoader()
    train_dataset = loader.load_data(limit=args.max_samples)
    
    if len(train_dataset) == 0:
        raise ValueError("No training data loaded.")
    logger.info(f"Loaded {len(train_dataset)} training samples.")

    # --- Load Model and Processor ---
    logger.info("===== LOADING MODEL AND PROCESSOR =====")
    model_load_path = args.local_model_path if args.local_model_path and os.path.exists(args.local_model_path) else args.model_name
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_load_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            model_load_path, 
            min_pixels=128*28*28, 
            max_pixels=256*28*28, 
            padding_side="right",
            trust_remote_code=True
        )
        logger.info("Model and processor loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model or processor: {e}")
        raise

    # --- Setup DataLoader with collate_fn ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=accelerator.device)
    )

    # --- Setup Optimizer ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # --- Prepare for distributed training ---
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # --- Training Setup ---
    stats_tracker = StatsTrackingCallback(run_name=f"deepspeed_{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    stats_tracker.start_training(args.max_steps)
    
    # --- Training Loop ---
    model.train()
    global_step = 0
    
    for epoch in range(10):  # We'll use max_steps as the main limiter
        for batch in train_loader:
            global_step += 1
            
            with accelerator.accumulate(model):
                inputs, labels = batch
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # Log progress
                if accelerator.is_local_main_process:
                    stats_tracker.log_step(global_step, loss.item(), args.learning_rate)
                    if global_step % args.logging_steps == 0:
                        logger.info(f"Epoch {epoch+1}, Step {global_step}: loss={loss.item():.6f}")
            
            # Check if we've reached max_steps
            if global_step >= args.max_steps:
                break
        
        # Check if we've reached max_steps after an epoch
        if global_step >= args.max_steps:
            break
    
    # --- End of Training ---
    stats_tracker.end_training(args, global_step)
    
    # --- Save Model ---
    logger.info("===== SAVING MODEL =====")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_local_main_process:
        save_dir = os.path.join(output_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)
        
        unwrapped_model.save_pretrained(
            save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            max_shard_size="20GB",
            state_dict=accelerator.get_state_dict(model),
        )
        
        processor.save_pretrained(save_dir)
        write_chat_template(processor, save_dir)
        logger.info(f"Model and processor saved to {save_dir}")
    
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()
