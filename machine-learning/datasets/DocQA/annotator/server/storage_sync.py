import boto3
import os
import json
import hashlib
from pathlib import Path
import time
import threading
import logging
import concurrent.futures
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('storage_sync')

class S3DatasetSync:
    """
    Provides direct interaction with S3 for dataset storage.
    Handles listing, downloading, and uploading dataset files.
    """

    def __init__(self, s3_bucket=None, prefix="datasets/", max_workers=10):
        """
        Initialize the S3 interface.

        Args:
            s3_bucket: S3 bucket name (if None, uses environment variable S3_BUCKET)
            prefix: S3 key prefix for all datasets
            max_workers: Maximum number of concurrent workers for S3 operations
        """
        self.s3_bucket = s3_bucket or os.environ.get("S3_BUCKET")
        self.prefix = prefix.rstrip('/') + '/' # Ensure prefix ends with /
        self.max_workers = max_workers
        self.temp_dir = Path("temp_s3_downloads") # For temporary local copies
        self.temp_dir.mkdir(exist_ok=True)

        # Configure transfer settings
        self.transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            max_concurrency=max_workers,
            multipart_chunksize=8 * 1024 * 1024,
            use_threads=True
        )

        # Initialize S3 client
        self.s3_client = None
        if self.s3_bucket:
            try:
                self.s3_client = boto3.client('s3')
                logger.info(f"S3 interface initialized with bucket: {self.s3_bucket}, prefix: {self.prefix}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {str(e)}")
                self.s3_client = None
        else:
            logger.warning("S3_BUCKET environment variable not set. S3 operations disabled.")

    def is_enabled(self):
        """Check if S3 interaction is enabled."""
        return self.s3_client is not None

    def _get_dataset_prefix(self, dataset_id):
        """Construct the S3 prefix for a specific dataset."""
        return f"{self.prefix}{dataset_id}/"

    def _get_pdf_prefix(self, dataset_id):
        """Construct the S3 prefix for PDFs within a dataset."""
        return f"{self._get_dataset_prefix(dataset_id)}pdfs/"

    def _get_vals_prefix(self, dataset_id):
        """Construct the S3 prefix for annotations within a dataset."""
        return f"{self._get_dataset_prefix(dataset_id)}vals/"

    def _get_annotation_key(self, dataset_id):
        """Construct the S3 key for the annotation file."""
        return f"{self._get_vals_prefix(dataset_id)}{dataset_id}.jsonl"

    def list_datasets(self):
        """List all dataset IDs available in S3 under the main prefix."""
        if not self.is_enabled():
            return []

        datasets = set()
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.prefix, Delimiter='/'):
                for prefix_info in page.get('CommonPrefixes', []):
                    # Extract dataset ID from common prefix (e.g., datasets/my_dataset/)
                    full_prefix = prefix_info.get('Prefix', '')
                    # Remove main prefix and trailing slash
                    dataset_id = full_prefix[len(self.prefix):].strip('/')
                    if dataset_id:
                        datasets.add(dataset_id)
            logger.info(f"Found remote datasets: {list(datasets)}")
            return sorted(list(datasets))
        except ClientError as e:
            logger.error(f"Error listing datasets in S3: {e}")
            # Handle specific errors like NoSuchBucket if needed
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.error(f"S3 bucket '{self.s3_bucket}' not found.")
            elif e.response['Error']['Code'] == 'AccessDenied':
                 logger.error(f"Access denied to S3 bucket '{self.s3_bucket}'. Check permissions.")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing datasets: {str(e)}")
            return []

    def list_pdfs_in_dataset(self, dataset_id):
        """List PDF filenames within a specific dataset in S3."""
        if not self.is_enabled():
            return []

        pdf_files = []
        pdf_prefix = self._get_pdf_prefix(dataset_id)
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=pdf_prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # Ensure it's a file directly under the pdfs/ prefix and ends with .pdf
                    if key != pdf_prefix and key.endswith('.pdf'):
                        filename = key[len(pdf_prefix):] # Get filename relative to pdfs/
                        if filename:
                            pdf_files.append(filename)
            logger.info(f"Found {len(pdf_files)} PDFs in dataset '{dataset_id}'")
            return sorted(pdf_files)
        except Exception as e:
            logger.error(f"Error listing PDFs for dataset {dataset_id}: {str(e)}")
            return []

    def download_annotation_content(self, dataset_id):
        """Download the content of the annotation file (.jsonl) from S3."""
        if not self.is_enabled():
            return None

        s3_key = self._get_annotation_key(dataset_id)
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            logger.info(f"Downloaded annotation file: {s3_key}")
            return content
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.info(f"Annotation file not found in S3: {s3_key}")
                return "" # Return empty string if file doesn't exist
            else:
                logger.error(f"Error downloading annotation file {s3_key}: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error downloading annotation file {s3_key}: {str(e)}")
            return None

    def upload_annotation_content(self, dataset_id, content):
        """Upload content to the annotation file (.jsonl) in S3."""
        if not self.is_enabled():
            return False

        s3_key = self._get_annotation_key(dataset_id)
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=content.encode('utf-8'),
                ContentType='application/jsonl'
            )
            logger.info(f"Uploaded annotation file: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading annotation file {s3_key}: {str(e)}")
            return False

    def download_pdf_to_temp(self, dataset_id, pdf_filename):
        """Download a specific PDF from S3 to a temporary local file."""
        if not self.is_enabled():
            return None

        s3_key = f"{self._get_pdf_prefix(dataset_id)}{pdf_filename}"
        # Create a unique temporary path
        temp_pdf_path = self.temp_dir / f"{dataset_id}_{pdf_filename}"

        try:
            logger.info(f"Downloading {s3_key} to {temp_pdf_path}...")
            self.s3_client.download_file(
                self.s3_bucket,
                s3_key,
                str(temp_pdf_path),
                Config=self.transfer_config
            )
            logger.info(f"Successfully downloaded PDF to temporary file: {temp_pdf_path}")
            return str(temp_pdf_path)
        except ClientError as e:
             if e.response['Error']['Code'] == 'NoSuchKey':
                 logger.error(f"PDF not found in S3: {s3_key}")
             else:
                 logger.error(f"Error downloading PDF {s3_key}: {e}")
             return None
        except Exception as e:
            logger.error(f"Unexpected error downloading PDF {s3_key}: {str(e)}")
            return None

    def upload_file(self, local_path, s3_key, content_type='application/pdf'):
        """Upload a local file to a specific S3 key."""
        if not self.is_enabled():
            return False

        try:
            logger.info(f"Uploading {local_path} to s3://{self.s3_bucket}/{s3_key}")
            self.s3_client.upload_file(
                local_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={'ContentType': content_type},
                Config=self.transfer_config
            )
            logger.info(f"Successfully uploaded: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading file {local_path} to {s3_key}: {str(e)}")
            return False

    def upload_bytes(self, data_bytes, s3_key, content_type='application/pdf'):
        """Upload bytes data to a specific S3 key."""
        if not self.is_enabled():
            return False
        try:
            logger.info(f"Uploading bytes data to s3://{self.s3_bucket}/{s3_key}")
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=data_bytes,
                ContentType=content_type
            )
            logger.info(f"Successfully uploaded bytes data: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading bytes data to {s3_key}: {str(e)}")
            return False

    def ensure_dataset_structure(self, dataset_id):
        """Ensure basic S3 'folders' (empty objects) exist for a dataset."""
        if not self.is_enabled():
            return

        prefixes_to_ensure = [
            self._get_dataset_prefix(dataset_id),
            self._get_pdf_prefix(dataset_id),
            self._get_vals_prefix(dataset_id)
        ]
        try:
            for prefix in prefixes_to_ensure:
                # Check if the prefix 'folder' object exists
                try:
                    self.s3_client.head_object(Bucket=self.s3_bucket, Key=prefix)
                except ClientError as e:
                    if e.response['Error']['Code'] == '404': # Not found
                         # Create an empty object to represent the folder
                         self.s3_client.put_object(Bucket=self.s3_bucket, Key=prefix, Body='')
                         logger.info(f"Created S3 placeholder for: {prefix}")
                    else:
                        raise # Re-raise other errors
        except Exception as e:
            logger.error(f"Error ensuring dataset structure for {dataset_id}: {str(e)}")

    def cleanup_temp_dir(self):
        """Remove temporary downloaded files."""
        try:
            for item in self.temp_dir.iterdir():
                item.unlink()
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}") 