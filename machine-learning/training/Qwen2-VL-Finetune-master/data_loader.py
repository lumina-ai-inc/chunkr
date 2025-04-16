import os
import io
import logging
from typing import List, Dict, Optional
from pathlib import Path
import concurrent.futures
import re

import boto3
from botocore.exceptions import ClientError
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value, Sequence
from dotenv import load_dotenv

# Assuming storage.py is accessible, adjust path if necessary
# If storage.py is in datasets/table-sophris/utils, you might need to adjust PYTHONPATH
# or copy/symlink storage.py here. For simplicity, we'll replicate minimal S3 listing.
# from storage import TableS3Storage # Ideal, but might cause import issues depending on structure

from models import Conversation, Message, ContentItem, TableTrainingSample

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simplified S3 interaction if TableS3Storage is not directly importable
class SimpleS3Fetcher:
    def __init__(self, bucket_name: str, dataset_name: str):
        self.bucket_name = bucket_name
        self.dataset_name = dataset_name
        self.s3_client = boto3.client('s3')
        self.base_prefix = f"{dataset_name}/" # Adjust if your base prefix is different
        self.images_prefix = f"{self.base_prefix}table_images/"
        self.html_prefix = f"{self.base_prefix}table_html/"

    def list_objects(self, prefix: str) -> List[str]:
        keys = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    keys.append(obj['Key'])
            logger.info(f"Found {len(keys)} objects with prefix '{prefix}'")
            return keys
        except Exception as e:
            logger.error(f"Error listing S3 objects with prefix '{prefix}': {e}")
            return []

    def download_file_content(self, key: str) -> Optional[bytes]:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"S3 object not found: {key}")
            else:
                logger.error(f"Error downloading S3 object {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading S3 object {key}: {str(e)}")
            return None

class TableDatasetLoader:
    def __init__(self, s3_bucket: Optional[str] = None, dataset_name: Optional[str] = None):
        load_dotenv(override=True)
        self.s3_bucket = s3_bucket or os.environ.get("S3_BUCKET")
        self.dataset_name = dataset_name or os.environ.get("DATASET_NAME")
        if not self.s3_bucket or not self.dataset_name:
            raise ValueError("S3 bucket and dataset name must be provided or set in environment variables.")

        self.s3_fetcher = SimpleS3Fetcher(self.s3_bucket, self.dataset_name)
        self.instruction = "OCR the table and convert it to HTML. Output the HTML directly in ```html``` tags."
        logger.info(f"Using instruction: '{self.instruction}'")

    def _get_table_id_from_key(self, key: str, prefix: str, suffix: str) -> str:
        filename = key[len(prefix):]
        return filename[:-len(suffix)] # Remove suffix (.jpg, .html)

    def _fetch_and_process_sample(self, image_key: str) -> Optional[TableTrainingSample]:
        table_id = self._get_table_id_from_key(image_key, self.s3_fetcher.images_prefix, ".jpg")
        html_key = f"{self.s3_fetcher.html_prefix}{table_id}.html"

        image_content = self.s3_fetcher.download_file_content(image_key)
        html_content_bytes = self.s3_fetcher.download_file_content(html_key)

        if image_content is None or html_content_bytes is None:
            logger.warning(f"Missing image or HTML for table_id: {table_id}")
            return None

        try:
            image = Image.open(io.BytesIO(image_content)).convert("RGB")
            html_content = html_content_bytes.decode('utf-8').strip()
            # Basic HTML structure validation (optional but recommended)
            if not (html_content.startswith('<table') or (html_content.startswith('```html') and '<table' in html_content)):
                 logger.warning(f"HTML content for {table_id} doesn't seem to start with a table or expected format. Content: {html_content[:100]}...")
                 # Decide if you want to skip these samples or proceed
                 # return None # Uncomment to skip

            # Clean HTML if needed (e.g., remove ```html tags if present)
            html_content = re.sub(r'^```html\s*', '', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'\s*```$', '', html_content)

            return TableTrainingSample(
                image=image,
                html=html_content,
                table_id=table_id
            )
        except Exception as e:
            logger.error(f"Error processing sample {table_id}: {e}")
            return None

    def load_data(self, limit: Optional[int] = None, max_workers: int = 10) -> Dataset:
        logger.info(f"Loading data from s3://{self.s3_bucket}/{self.dataset_name}")
        image_keys = self.s3_fetcher.list_objects(self.s3_fetcher.images_prefix)
        image_keys = [key for key in image_keys if key.lower().endswith(".jpg")] # Filter for jpg

        if limit:
            image_keys = image_keys[:limit]
            logger.info(f"Limiting dataset to {limit} samples.")

        logger.info(f"Found {len(image_keys)} table images. Fetching samples...")

        processed_samples: List[TableTrainingSample] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(self._fetch_and_process_sample, key): key for key in image_keys}
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    if result and isinstance(result.image, Image.Image): # Ensure image is a PIL Image
                        processed_samples.append(result)
                    elif result is None:
                         logger.warning(f"Sample processing returned None for key {key}")
                    elif not isinstance(result.image, Image.Image):
                         logger.warning(f"Sample processing resulted in invalid image type ({type(result.image)}) for key {key}")

                except Exception as exc:
                    logger.error(f"S3 key {key} generated an exception during processing: {exc}")

        if not processed_samples:
            logger.warning("No samples were successfully processed.")
            # Define the expected nested structure for an empty dataset (no top-level image)
            empty_features = Features({
                'messages': Sequence({
                    'role': Value('string'),
                    'content': Sequence({
                        'type': Value('string'),
                        'text': Value('string'),
                        'image': HFImage() # Use HF Image type for schema
                    })
                })
            })
            # Provide empty list only for messages key
            return Dataset.from_dict({"messages": []}, features=empty_features)

        logger.info(f"Successfully processed {len(processed_samples)} samples.")

        # Convert directly to the required dictionary format for HF Dataset
        hf_data = []
        for sample in processed_samples:
            if not isinstance(sample.image, Image.Image):
                 logger.warning(f"Skipping sample {sample.table_id} during final conversion due to invalid image type: {type(sample.image)}")
                 continue

            # Create the dictionary structure *without* the top-level 'image' key
            conversation_dict = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.instruction, "image": None},
                            # Keep the nested image reference
                            {"type": "image", "image": sample.image, "text": None}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": sample.html, "image": None}
                        ]
                    }
                ]
            }
            hf_data.append(conversation_dict)
        
        if not hf_data:
             logger.error("No valid samples could be converted to the dataset format after final checks.")
             # Return empty dataset matching the simplified structure
             empty_features = Features({
                 'messages': Sequence({
                     'role': Value('string'),
                     'content': Sequence({
                         'type': Value('string'),
                         'text': Value('string'),
                         'image': HFImage()
                     })
                 })
             })
             return Dataset.from_dict({"messages": []}, features=empty_features)

        # Define Hugging Face Dataset features explicitly, *without* the top-level 'image'
        features = Features({
            'messages': Sequence({ # List of messages
                'role': Value('string'),
                'content': Sequence({ # List of content items per message
                    'type': Value('string'),
                    'text': Value('string'), # Allow null text
                    'image': HFImage()      # Feature for the nested image reference
                })
            })
        })

        # Create Hugging Face Dataset from the list of dictionaries
        hf_dataset = Dataset.from_list(hf_data, features=features)

        logger.info(f"Created Hugging Face Dataset with {len(hf_dataset)} samples.")
        return hf_dataset

if __name__ == '__main__':
    # Example usage:
    loader = TableDatasetLoader()
    dataset = loader.load_data(limit=10) # Load 10 samples
    print(dataset)
    if len(dataset) > 0:
        print("\nFirst sample:")
        print(dataset[0])
        # Accessing the image requires handling the HF Dataset format
        try:
            first_image = dataset[0]['messages'][0]['content'][1]['image']
            print(f"\nFirst image type: {type(first_image)}")
            if isinstance(first_image, Image.Image):
                 print(f"First image size: {first_image.size}")
            # first_image.show() # Uncomment to display the first image if running locally
        except (IndexError, KeyError, TypeError) as e:
            print(f"Could not access image in first sample: {e}")
