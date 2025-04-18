import os
import io
import json
import logging
import argparse
import random
import base64
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
import time

# Assuming data_loader.py and its dependencies are accessible
from data_loader import TableDatasetLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_for_saving(sample, instruction):
    """Formats a raw sample for saving to JSONL, converting image to bytes."""
    if not isinstance(sample.image, Image.Image):
        logger.warning(f"Sample {sample.table_id} has invalid image type: {type(sample.image)}. Skipping.")
        return None

    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        # Use PNG as a lossless format, adjust if needed
        sample.image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Convert bytes to base64 string for JSON serialization
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        logger.warning(f"Failed to convert image to base64 for sample {sample.table_id}: {e}. Skipping.")
        return None

    html_content = f"```html\n{sample.html}\n```" # Wrap HTML

    # Construct the messages list, using base64-encoded image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_base64", "image_base64": img_base64}  # Store as base64 string
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": html_content}
            ]
        }
    ]
    # Return the structure ready for JSONL
    return {"messages": messages, "table_id": sample.table_id} # Keep ID for debugging

def fetch_sample_worker(args):
    """Worker function for parallel sample fetching"""
    loader, key, retry_count = args
    
    for attempt in range(retry_count + 1):
        try:
            sample = loader._fetch_and_process_sample(key)
            if sample and isinstance(sample.image, Image.Image):
                return sample, None
            return None, f"Invalid sample or image from key '{key}'"
        except Exception as e:
            if attempt < retry_count:
                time.sleep(1)  # Wait before retry
                continue
            return None, f"Error processing key '{key}': {str(e)}"

# Helper for parallel formatting in save_split
def format_sample_worker(args):
    sample, instruction = args
    return format_for_saving(sample, instruction)

def main():
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Prepare dataset for table OCR training")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for dataset files")
    parser.add_argument("--data_limit", type=int, default=None, help="Limit number of total samples")
    parser.add_argument("--s3_bucket", type=str, default=None, help="S3 bucket (overrides .env)")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name in S3 (overrides .env)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data for validation")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--parallel", type=int, default=16, help="Number of parallel workers for fetching/formatting (increased default)")
    parser.add_argument("--retry", type=int, default=2, help="Number of retries for failed fetches")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for processing samples (increased default)")
    args = parser.parse_args()

    # Get dataset name from args or environment
    s3_bucket = args.s3_bucket
    dataset_name = args.dataset_name or os.getenv("DATASET_NAME")
    if not dataset_name:
        logger.error("Dataset name not specified. Please set DATASET_NAME in .env or use --dataset_name")
        return
    
    data_limit = args.data_limit
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    max_workers = args.parallel
    retry_count = args.retry
    batch_size = args.batch_size
    
    # Create dataset-specific output directory
    safe_dataset_name = dataset_name.replace("/", "_").replace("\\", "_")
    dataset_dir = os.path.join(output_dir, safe_dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save the dataset name for other scripts
    with open(os.path.join(dataset_dir, "dataset_name.txt"), "w") as f:
        f.write(safe_dataset_name)

    logger.info(f"Preparing dataset: {safe_dataset_name}")
    logger.info(f"Output directory: {dataset_dir}")
    
    # Ensure splits sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.warning(f"Split ratios sum to {total_ratio}, normalizing to 1.0")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output files with dataset name and limit info
    train_file = os.path.join(dataset_dir, f"train.jsonl")
    val_file = os.path.join(dataset_dir, f"val.jsonl")
    test_file = os.path.join(dataset_dir, f"test.jsonl")
    
    # Save the split information
    with open(os.path.join(dataset_dir, "dataset_info.json"), "w") as f:
        json.dump({
            "dataset_name": dataset_name,
            "train_file": train_file,
            "val_file": val_file,
            "test_file": test_file,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": args.seed,
            "data_limit": data_limit
        }, f, indent=2)

    # --- Load Raw Data ---
    logger.info(f"Loading raw dataset samples from '{dataset_name}'...")
    loader = TableDatasetLoader(s3_bucket=s3_bucket, dataset_name=dataset_name)
    raw_samples = []
    
    # List all image keys
    logger.info("Listing objects in S3 bucket...")
    image_keys = loader.s3_fetcher.list_objects(loader.s3_fetcher.images_prefix)
    image_keys = [key for key in image_keys if key.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not image_keys:
        logger.error(f"No image keys found in s3://{loader.s3_bucket}/{loader.s3_fetcher.images_prefix}")
        return

    target_keys = image_keys
    if data_limit:
        random.shuffle(target_keys)
        target_keys = target_keys[:data_limit]
        logger.info(f"Limiting dataset to {data_limit} samples.")

    logger.info(f"Found {len(target_keys)} table images. Fetching samples using {max_workers} parallel workers...")
    
    # Process samples in batches to show progress and avoid memory issues
    valid_raw_samples = 0
    skipped_samples = 0
    
    # Process in batches with a progress bar
    for batch_start in range(0, len(target_keys), batch_size):
        batch_end = min(batch_start + batch_size, len(target_keys))
        batch_keys = target_keys[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(target_keys) + batch_size - 1)//batch_size} ({batch_end}/{len(target_keys)} samples)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create tasks with loader and retry info
            tasks = [(loader, key, retry_count) for key in batch_keys]
            results = list(tqdm(executor.map(fetch_sample_worker, tasks), total=len(tasks), desc="Fetching samples"))
        
        # Process results from this batch
        batch_valid = 0
        batch_skipped = 0
        
        for sample, error in results:
            if sample:
                raw_samples.append(sample)
                batch_valid += 1
            else:
                if error:
                    logger.debug(error)  # Only log details at debug level to reduce output spam
                batch_skipped += 1
        
        valid_raw_samples += batch_valid
        skipped_samples += batch_skipped
        
        logger.info(f"Batch complete: {batch_valid} valid samples, {batch_skipped} skipped")
        
        # Save progress checkpoint to allow resuming in case of failure
        if valid_raw_samples > 0:
            checkpoint_file = os.path.join(dataset_dir, "checkpoint.json")
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "processed": batch_end,
                    "total": len(target_keys),
                    "valid_samples": valid_raw_samples,
                    "skipped_samples": skipped_samples
                }, f)

    logger.info(f"Loaded {valid_raw_samples} valid raw samples. Skipped {skipped_samples}.")
    if not raw_samples:
        logger.error("No valid raw samples loaded. Exiting.")
        return

    # --- Split the data into train, validation, and test sets ---
    random.shuffle(raw_samples)
    
    train_size = int(len(raw_samples) * train_ratio)
    val_size = int(len(raw_samples) * val_ratio)
    
    train_samples = raw_samples[:train_size]
    val_samples = raw_samples[train_size:train_size+val_size]
    test_samples = raw_samples[train_size+val_size:]
    
    logger.info(f"Split dataset into {len(train_samples)} training, {len(val_samples)} validation, and {len(test_samples)} test samples")
    
    # --- Save each split to its own file (with parallel formatting) ---
    def save_split(samples, output_file, instruction, num_workers):
        saved_count = 0
        skipped_count = 0
        format_errors = 0

        with open(output_file, 'w', encoding='utf-8') as f, \
             concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

            # Prepare tasks for formatting
            format_tasks = [(sample, instruction) for sample in samples]

            # Process formatting in parallel
            formatted_results = list(tqdm(executor.map(format_sample_worker, format_tasks),
                                          total=len(samples),
                                          desc=f"Formatting {os.path.basename(output_file)}"))

            # Write formatted results sequentially (safer for single file)
            for formatted_data in tqdm(formatted_results, desc=f"Writing {os.path.basename(output_file)}"):
                if formatted_data:
                    try:
                        json_record = json.dumps(formatted_data)
                        f.write(json_record + '\n')
                        saved_count += 1
                    except Exception as e:
                        # Log specific error if needed, using table_id from formatted_data
                        # logger.error(f"Error saving sample {formatted_data.get('table_id', 'UNKNOWN')}: {e}")
                        format_errors += 1
                else:
                    # This count includes samples skipped by format_for_saving
                    skipped_count += 1

        if format_errors > 0:
             logger.warning(f"Encountered {format_errors} errors during JSON serialization for {os.path.basename(output_file)}")

        return saved_count, skipped_count + format_errors # Combine skipped and serialization errors

    # Save train split
    logger.info(f"Saving training samples to {train_file}...")
    train_saved, train_skipped = save_split(train_samples, train_file, loader.instruction, max_workers)
    
    # Save validation split
    logger.info(f"Saving validation samples to {val_file}...")
    val_saved, val_skipped = save_split(val_samples, val_file, loader.instruction, max_workers)
    
    # Save test split
    logger.info(f"Saving test samples to {test_file}...")
    test_saved, test_skipped = save_split(test_samples, test_file, loader.instruction, max_workers)
    
    # Report results
    logger.info("=== Dataset Preparation Complete ===")
    logger.info(f"Dataset: {safe_dataset_name}")
    logger.info(f"Training set: {train_saved} samples saved ({train_skipped} skipped)")
    logger.info(f"Validation set: {val_saved} samples saved ({val_skipped} skipped)")
    logger.info(f"Test set: {test_saved} samples saved ({test_skipped} skipped)")
    logger.info(f"Total samples across all splits: {train_saved + val_saved + test_saved}")
    
    # Make sure this exact line is always printed so run.sh can find it
    print(f"Output directory: {dataset_dir}")
    logger.info(f"Output directory: {dataset_dir}")
    
    # Also write the dataset path to a file that can be easily read by other scripts
    with open(os.path.join(output_dir, "dataset_path.txt"), "w") as f:
        f.write(dataset_dir)
    
    # Create a metadata file with table_ids in each split for reference
    with open(os.path.join(dataset_dir, "split_metadata.json"), "w") as f:
        json.dump({
            "dataset_name": dataset_name,
            "data_limit": data_limit,
            "train_ids": [s.table_id for s in train_samples if hasattr(s, 'table_id')],
            "val_ids": [s.table_id for s in val_samples if hasattr(s, 'table_id')],
            "test_ids": [s.table_id for s in test_samples if hasattr(s, 'table_id')]
        }, f, indent=2)
    
    # Create a symlink from the default data dir to make it easier to use with trainer
    if dataset_dir != os.path.join(output_dir, "default"):
        default_dir = os.path.join(output_dir, "default")
        if os.path.exists(default_dir) or os.path.islink(default_dir):
            try:
                os.remove(default_dir)
            except:
                logger.warning(f"Could not remove existing default symlink at {default_dir}")
        
        try:
            # Create relative symlink
            os.symlink(os.path.relpath(dataset_dir, output_dir), default_dir)
            logger.info(f"Created default symlink at {default_dir} -> {dataset_dir}")
        except:
            logger.warning(f"Could not create default symlink at {default_dir}")

    # Delete checkpoint file now that we're done
    checkpoint_file = os.path.join(dataset_dir, "checkpoint.json")
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
        except:
            pass

if __name__ == "__main__":
    main() 