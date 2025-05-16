import os
import json
import logging
import requests
from pathlib import Path
from typing import Optional, Dict, List, Any
import concurrent.futures
from .storage import TableS3Storage
from chunkr_ai import Chunkr
from chunkr_ai.models import Configuration, SegmentProcessing, TaskResponse
from chunkr_ai.models import (
    GenerationConfig,
    SegmentType,
    CroppingStrategy,
)
import argparse
import time
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from .pdf_queue import PDFQueue
import redis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('chunkr_processor')

class ChunkrProcessor:
    """
    Processes datasheet PDFs with Chunkr API and stores the results in S3.
    Handles table extraction and related outputs.
    """

    def __init__(self, api_key=None, s3_bucket=None, dataset_name=None, max_workers=5):
        """
        Initialize the Chunkr processor.

        Args:
            api_key: Chunkr API key (if None, uses environment variable CHUNKR_API_KEY)
            s3_bucket: S3 bucket name (if None, uses environment variable S3_BUCKET)
            dataset_name: Dataset name (if None, uses environment variable DATASET_NAME)
            max_workers: Maximum number of concurrent workers for processing
        """
        load_dotenv(override=True)
        self.api_key = os.environ.get("CHUNKR_API_KEY")
        if not self.api_key:
            logger.warning("No Chunkr API key provided. Set CHUNKR_API_KEY environment variable or pass in constructor.")
        
        self.chunkr = Chunkr(api_key=self.api_key)
        self.s3_storage = TableS3Storage(s3_bucket, dataset_name)
        self.max_workers = max_workers
        
        # Define paths for different outputs in S3
        base_prefix = f"{self.s3_storage.base_prefix}{self.s3_storage.dataset_name}/"
        
        # Source of PDFs (created by DatasheetDownloader)
        self.raw_pdfs_prefix = f"{base_prefix}raw-pdfs/"
        
        # Output directories
        self.pdfs_prefix = f"{base_prefix}pdfs/"
        self.chunkr_outputs_prefix = f"{base_prefix}chunkr_outputs/"
        self.tables_prefix = f"{base_prefix}tables/"
        self.table_mkd_prefix = f"{base_prefix}table_mkd/"
        self.table_html_prefix = f"{base_prefix}table_html/"
        
        # Local temporary directories
        self.temp_dir = Path("temp_chunkr_processing")
        self.temp_dir.mkdir(exist_ok=True, parents=True)

    def _ensure_output_structure(self):
        """Ensure all necessary folder structures exist in S3."""
        if not self.s3_storage.is_enabled():
            return False
        
        prefixes = [
            self.pdfs_prefix,
            self.chunkr_outputs_prefix,
            self.tables_prefix,
            self.table_mkd_prefix,
            self.table_html_prefix,
        ]
        
        success = True
        for prefix in prefixes:
            try:
                self.s3_storage.s3_client.put_object(
                    Bucket=self.s3_storage.s3_bucket,
                    Key=prefix,
                    Body=''
                )
                logger.info(f"Ensured S3 folder structure: {prefix}")
            except Exception as e:
                logger.error(f"Error ensuring S3 folder structure {prefix}: {str(e)}")
                success = False
        
        return success

    def _process_pdf_with_chunkr(self, pdf_path: Path) -> Optional[TaskResponse]:
        """
        Send a PDF to Chunkr API for processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict containing Chunkr's response or None if failed
        """
        if not self.api_key:
            logger.error("Cannot process PDF: No Chunkr API key provided")
            return None
        
        try:
            logger.info(f"Sending PDF to Chunkr API: {pdf_path}")
            
            # Configure Chunkr processing
            config = Configuration(
                chunk_processing={
                    "ignore_headers_and_footers": False
                },
                segment_processing=SegmentProcessing(
                    Table=GenerationConfig(
                        html="Auto",
                        markdown="Auto",
                        crop_image=CroppingStrategy.ALL,
                    ),
                )
            )
            
            # Upload and process document
            task: TaskResponse  = self.chunkr.upload(str(pdf_path), config)
            result=task
            
            logger.info(f"Successfully processed {pdf_path.name} with Chunkr")
            return result
                
        except Exception as e:
            logger.error(f"Error processing PDF with Chunkr: {str(e)}")
            return None

    def _download_image_data(self, image_url: str) -> Optional[bytes]:
        """Downloads image data from a URL."""
        if not image_url:
            return None
        try:
            response = requests.get(image_url, timeout=30) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logger.info(f"Successfully downloaded image data from {image_url}")
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image data from {image_url}: {e}")
            return None

    def _extract_tables_from_response(self, task: TaskResponse) -> List[Dict]:
        """
        Extract table information from Chunkr response, including image data.

        Args:
            task: Chunkr TaskResponse object.

        Returns:
            List of table information dictionaries, each potentially including 'image_data'.
        """
        tables = []
        if not task or not task.output or not task.output.chunks:
             logger.warning("No chunks found in Chunkr response.")
             return tables

        try:
            for chunk in task.output.chunks:
                if not chunk.segments:
                    continue
                for segment in chunk.segments:
                    # Use SegmentType enum for comparison
                    if segment.segment_type == SegmentType.TABLE:
                        table_dict = segment.model_dump() # Convert segment to dict
                        image_url = segment.image

                        if image_url:
                            image_data = self._download_image_data(image_url)
                            if image_data:
                                table_dict['image_data'] = image_data
                                tables.append(table_dict)
                            else:
                                # Log error if download failed, but might still append table info without image?
                                # Or skip entirely if image is critical? Let's skip for now.
                                logger.error(f"Failed to download image for table segment {segment.segment_id}. Skipping this table.")
                        else:
                            # Log error if the image URL itself is missing
                            logger.error(f"Table segment {segment.segment_id} is missing image URL. Skipping this table.")

            logger.info(f"Extracted {len(tables)} tables with image data from Chunkr response")
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables from response: {str(e)}")
            import traceback
            logger.error(traceback.format_exc()) # Log full traceback for debugging
            return []

    def _save_json_to_s3(self, data: Dict, s3_key: str) -> bool:
        """
        Save JSON data to S3.
        
        Args:
            data: Data to save as JSON
            s3_key: S3 key for the file
            
        Returns:
            bool: Success or failure
        """
        if not self.s3_storage.is_enabled():
            return False
        
        try:
            json_data = json.dumps(data, indent=2)
            self.s3_storage.s3_client.put_object(
                Bucket=self.s3_storage.s3_bucket,
                Key=s3_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            logger.info(f"Saved JSON to S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON to S3 {s3_key}: {str(e)}")
            return False

    def _save_text_to_s3(self, text: str, s3_key: str, content_type: str) -> bool:
        """
        Save text data to S3.
        
        Args:
            text: Text data to save
            s3_key: S3 key for the file
            content_type: MIME type of the content
            
        Returns:
            bool: Success or failure
        """
        if not self.s3_storage.is_enabled():
            return False
        
        try:
            self.s3_storage.s3_client.put_object(
                Bucket=self.s3_storage.s3_bucket,
                Key=s3_key,
                Body=text.encode('utf-8'),
                ContentType=content_type
            )
            logger.info(f"Saved text to S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error saving text to S3 {s3_key}: {str(e)}")
            return False

    def _save_bytes_to_s3(self, data: bytes, s3_key: str, content_type: str) -> bool:
        """
        Save byte data to S3.

        Args:
            data: Byte data to save.
            s3_key: S3 key for the file.
            content_type: MIME type of the content.

        Returns:
            bool: Success or failure.
        """
        if not self.s3_storage.is_enabled():
            return False

        try:
            self.s3_storage.s3_client.put_object(
                Bucket=self.s3_storage.s3_bucket,
                Key=s3_key,
                Body=data,
                ContentType=content_type
            )
            logger.info(f"Saved bytes to S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error saving bytes to S3 {s3_key}: {str(e)}")
            return False

    def _copy_pdf_to_processed_folder(self, source_key: str, filename: str) -> bool:
        """
        Copy a PDF from raw-pdfs to the pdfs folder in S3.
        
        Args:
            source_key: Source S3 key of the PDF
            filename: Name of the PDF file
            
        Returns:
            bool: Success or failure
        """
        if not self.s3_storage.is_enabled():
            return False
        
        destination_key = f"{self.pdfs_prefix}{filename}"
        
        try:
            # Copy the object
            copy_source = {'Bucket': self.s3_storage.s3_bucket, 'Key': source_key}
            self.s3_storage.s3_client.copy(
                copy_source, 
                self.s3_storage.s3_bucket,
                destination_key
            )
            logger.info(f"Copied PDF from {source_key} to {destination_key}")
            return True
        except Exception as e:
            logger.error(f"Error copying PDF to processed folder: {str(e)}")
            return False

    def _check_outputs_exist(self, base_filename: str) -> bool:
        """
        Check if outputs for this PDF already exist in S3.
        
        Args:
            base_filename: Base filename of the PDF (without extension)
            
        Returns:
            bool: True if outputs exist, False otherwise
        """
        if not self.s3_storage.is_enabled():
            return False
        
        # Check if Chunkr output JSON exists
        output_key = f"{self.chunkr_outputs_prefix}{base_filename}.json"
        
        try:
            self.s3_storage.s3_client.head_object(
                Bucket=self.s3_storage.s3_bucket,
                Key=output_key
            )
            logger.info(f"Outputs already exist for {base_filename}")
            return True
        except Exception:
            return False

    def process_pdf_from_s3(self, pdf_key: str) -> bool:
        """
        Process a PDF from S3 with Chunkr and store all outputs.
        
        Args:
            pdf_key: S3 key of the PDF file
            
        Returns:
            bool: Success or failure
        """
        if not self.s3_storage.is_enabled():
            return False
        
        # Extract filename and base name
        filename = pdf_key.split('/')[-1]
        base_filename = Path(filename).stem
        
        # Check if outputs already exist
        if self._check_outputs_exist(base_filename):
            logger.info(f"Skipping {base_filename} as outputs already exist")
            return True
        
        # Download PDF to temp directory
        local_pdf_path = self.temp_dir / filename
        try:
            self.s3_storage.s3_client.download_file(
                self.s3_storage.s3_bucket,
                pdf_key,
                str(local_pdf_path)
            )
            logger.info(f"Downloaded {filename} from S3 for processing")
            
            # Process PDF with Chunkr
            chunkr_response_task = self._process_pdf_with_chunkr(local_pdf_path)
            
            # Clean up temporary file
            local_pdf_path.unlink(missing_ok=True)
            
            if not chunkr_response_task:
                return False
            
            # Copy PDF to processed folder
            self._copy_pdf_to_processed_folder(pdf_key, filename)
            
            # Save main Chunkr output
            main_output_key = f"{self.chunkr_outputs_prefix}{base_filename}.json"
            self._save_json_to_s3(chunkr_response_task.model_dump(mode='json'), main_output_key)
            
            # Extract and save tables
            tables = self._extract_tables_from_response(chunkr_response_task)
            
            for i, table in enumerate(tables):
                table_id = table.get('id', i)
                if 'segment_id' in table and not table_id:
                    table_id = table['segment_id']
                table_filename_base = f"{base_filename}_{table_id}"
                
                # Save table JSON (excluding the large image data)
                table_json_to_save = {k: v for k, v in table.items() if k != 'image_data'}
                table_json_key = f"{self.tables_prefix}{table_filename_base}.json"
                self._save_json_to_s3(table_json_to_save, table_json_key)
                
                # Save table markdown if available
                if 'markdown' in table and table['markdown']:
                    table_mkd_key = f"{self.table_mkd_prefix}{table_filename_base}.md"
                    self._save_text_to_s3(table['markdown'], table_mkd_key, 'text/markdown')
                
                # Save table HTML if available
                if 'html' in table and table['html']:
                    table_html_key = f"{self.table_html_prefix}{table_filename_base}.html"
                    self._save_text_to_s3(table['html'], table_html_key, 'text/html')
                
                # Save table image if available
                if 'image_data' in table and table['image_data']:
                    # Define image prefix if not already done in __init__
                    if not hasattr(self, 'table_images_prefix'):
                        self.table_images_prefix = f"{self.s3_storage.base_prefix}{self.s3_storage.dataset_name}/table_images/"
                        # Optional: Ensure this prefix exists via _ensure_output_structure
                        # self._ensure_output_structure() # Call again or modify it

                    table_image_key = f"{self.table_images_prefix}{table_filename_base}.jpg" # Assuming jpg
                    self._save_bytes_to_s3(table['image_data'], table_image_key, 'image/jpeg')
            
            logger.info(f"Successfully processed {base_filename} and stored all outputs")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {filename} from S3: {str(e)}")
            import traceback
            logger.error(traceback.format_exc()) # Log full traceback
            local_pdf_path.unlink(missing_ok=True)
            return False

    def process_pdf_wrapper(self, pdf_key, api_key, s3_bucket, dataset_name):
        """
        Standalone function for processing a PDF that can be safely pickled for multiprocessing.
        
        Args:
            pdf_key: S3 key of the PDF file
            api_key: Chunkr API key
            s3_bucket: S3 bucket name
            dataset_name: Dataset name
            
        Returns:
            tuple: (pdf_key, success, skipped) indicating the processing result
        """
        # Create a new processor instance in this process
        processor = ChunkrProcessor(api_key=api_key, s3_bucket=s3_bucket, dataset_name=dataset_name)
        
        # Extract filename and base name for checking if already processed
        filename = pdf_key.split('/')[-1]
        base_filename = Path(filename).stem
        
        # Check if outputs already exist
        if processor._check_outputs_exist(base_filename):
            return pdf_key, True, True  # Key, success, skipped
        
        # Process the PDF
        result = processor.process_pdf_from_s3(pdf_key)
        return pdf_key, result, False  # Key, success, not skipped

    def process_datasheets_from_s3(self, limit: Optional[int] = None) -> Dict[str, int]:
        """
        Process datasheet PDFs from the S3 raw-pdfs folder created by DatasheetDownloader.
        Uses multiprocessing for parallelism.

        Args:
            limit: Optional limit on number of PDFs to process

        Returns:
            Dict with processing statistics
        """
        if not self.s3_storage.is_enabled():
            return {"error": "S3 storage not enabled"}

        # Ensure output directories exist
        self._ensure_output_structure()

        # List PDFs in raw-pdfs folder
        pdfs = []
        try:
            paginator = self.s3_storage.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.s3_storage.s3_bucket, Prefix=self.raw_pdfs_prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.pdf'):
                        pdfs.append(obj['Key'])
        except Exception as e:
            logger.error(f"Error listing PDFs in S3: {str(e)}")
            return {"error": str(e)}

        if limit:
            pdfs = pdfs[:limit]

        logger.info(f"Found {len(pdfs)} PDFs to process from S3 raw-pdfs folder using {self.max_workers} processes.")

        # Process PDFs
        stats = {
            "total": len(pdfs),
            "success": 0,
            "skipped": 0,
            "failed": 0
        }

        # Create arguments for the worker function
        process_args = [(pdf_key, self.api_key, self.s3_storage.s3_bucket, self.s3_storage.dataset_name) 
                         for pdf_key in pdfs]
                         
        # Use ProcessPoolExecutor with the standalone function
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Map the standalone function to the process arguments
            for i, (pdf_key, success, skipped) in enumerate(executor.map(self.process_pdf_wrapper, *zip(*process_args))):
                processed_count = i + 1
                total_to_process = len(pdfs)
                base_filename = Path(pdf_key.split('/')[-1]).stem
                
                if skipped:
                    stats["skipped"] += 1
                    logger.info(f"({processed_count}/{total_to_process}) Confirmed skip: {base_filename}")
                elif success:
                    stats["success"] += 1
                    logger.info(f"({processed_count}/{total_to_process}) Success: {base_filename}")
                else:
                    stats["failed"] += 1
                    logger.warning(f"({processed_count}/{total_to_process}) Failed: {base_filename}")

        return stats

    def process_document(self, pdf_path: str, output_dir: Optional[str] = None) -> bool:
        """
        Process a local PDF document with Chunkr and store all outputs in S3.
        This is a convenience method for processing individual files directly.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional local output directory (for testing)
            
        Returns:
            bool: Success or failure
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        # Ensure S3 structure
        self._ensure_output_structure()
        
        # Get base filename (without extension)
        base_filename = pdf_path.stem
        
        # Check if outputs already exist
        if self._check_outputs_exist(base_filename):
            logger.info(f"Skipping {base_filename} as outputs already exist")
            return True
        
        # Process PDF with Chunkr
        chunkr_response_task = self._process_pdf_with_chunkr(pdf_path)
        if not chunkr_response_task:
            return False
        
        # Upload PDF to processed folder
        pdf_key = f"{self.pdfs_prefix}{pdf_path.name}"
        try:
            with open(pdf_path, 'rb') as file:
                self.s3_storage.s3_client.upload_fileobj(
                    file,
                    self.s3_storage.s3_bucket,
                    pdf_key,
                    ExtraArgs={
                        'ContentType': 'application/pdf'
                    },
                    Config=self.s3_storage.transfer_config
                )
            logger.info(f"Uploaded PDF to S3: {pdf_key}")
        except Exception as e:
            logger.error(f"Error uploading PDF to S3: {str(e)}")
            return False
        
        # Save main Chunkr output
        main_output_key = f"{self.chunkr_outputs_prefix}{base_filename}.json"
        self._save_json_to_s3(chunkr_response_task.model_dump(mode='json'), main_output_key)
        
        # Extract and save tables
        tables = self._extract_tables_from_response(chunkr_response_task)
        
        for i, table in enumerate(tables):
            table_id = table.get('id', i)
            if 'segment_id' in table and not table_id:
                table_id = table['segment_id']
            table_filename_base = f"{base_filename}_{table_id}"
            
            # Save table JSON (excluding image data)
            table_json_to_save = {k: v for k, v in table.items() if k != 'image_data'}
            table_json_key = f"{self.tables_prefix}{table_filename_base}.json"
            self._save_json_to_s3(table_json_to_save, table_json_key)
            
            # Save table markdown if available
            if 'markdown' in table and table['markdown']:
                table_mkd_key = f"{self.table_mkd_prefix}{table_filename_base}.md"
                self._save_text_to_s3(table['markdown'], table_mkd_key, 'text/markdown')
            
            # Save table HTML if available
            if 'html' in table and table['html']:
                table_html_key = f"{self.table_html_prefix}{table_filename_base}.html"
                self._save_text_to_s3(table['html'], table_html_key, 'text/html')
            
            # Save table image if available
            if 'image_data' in table and table['image_data']:
                # Define image prefix if not already done in __init__
                if not hasattr(self, 'table_images_prefix'):
                    self.table_images_prefix = f"{self.s3_storage.base_prefix}{self.s3_storage.dataset_name}/table_images/"
                    # Optional: Ensure this prefix exists via _ensure_output_structure
                    # self._ensure_output_structure() # Call again or modify it

                table_image_key = f"{self.table_images_prefix}{table_filename_base}.jpg" # Assuming jpg
                self._save_bytes_to_s3(table['image_data'], table_image_key, 'image/jpeg')
        
        # Save locally too if requested (saving the full response including image data might be large)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            # Save the main JSON response (consider excluding image data if files get too big)
            local_main_output = chunkr_response_task.model_dump(mode='json')
            with open(output_path / f"{base_filename}_main_response.json", 'w') as f:
                json.dump(local_main_output, f, indent=2)
            logger.info(f"Saved local main response JSON to {output_path / f'{base_filename}_main_response.json'}")

            # Optionally save individual table JSONs and images locally too
            # ... (similar logic as S3 saving but writing to local files) ...

        logger.info(f"Successfully processed {base_filename} and stored all outputs")
        return True

    def cleanup(self):
        """Clean up temporary files."""
        try:
            # Use shutil.rmtree for potentially faster deletion if the directory exists
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            else:
                logger.info(f"Temporary directory already removed: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")

    async def process_file(self, file_path: str, config: Configuration) -> Optional[TaskResponse]:
        """
        Process a single PDF file and return the TaskResponse.
        
        Args:
            file_path: Path to the PDF file
            config: Chunkr configuration
            
        Returns:
            TaskResponse if successful, None if failed
        """
        try:
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            result = self._process_pdf_with_chunkr(pdf_path)
            return result
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None


def process_pdf_from_queue():
    """
    Process a single PDF from the queue.
    
    Returns:
        dict: Result of processing or None if no task or error
    """
    try:
        # Import here to avoid circular imports
        from pdf_queue import check_redis_running, start_redis_server
        
        # Ensure Redis is running
        if not check_redis_running():
            logger.warning("Redis is not running, attempting to start...")
            if not start_redis_server():
                logger.error("Failed to start Redis server. Cannot process queue.")
                time.sleep(5)  # Wait before retry
                return None
            
        queue = PDFQueue()
        
        # Get a task from the queue
        task = queue.get_task(timeout=5)
        
        if not task:
            logger.info("No tasks in queue")
            return None
        
        # Extract task data
        task_id = task["task_id"]
        pdf_key = task["pdf_key"]
        api_key = task["api_key"]
        s3_bucket = task["s3_bucket"]
        dataset_name = task["dataset_name"]
        
        logger.info(f"Processing task {task_id} for PDF: {pdf_key}")
        
        # Create a processor for this task
        processor = ChunkrProcessor(
            api_key=api_key,
            s3_bucket=s3_bucket,
            dataset_name=dataset_name
        )
        
        # Extract filename and base name for checking if already processed
        filename = pdf_key.split('/')[-1]
        base_filename = Path(filename).stem
        
        # Check if outputs already exist
        if processor._check_outputs_exist(base_filename):
            logger.info(f"Skipping {base_filename} as outputs already exist")
            queue.complete_task(task_id, success=True, skipped=True)
            return {"task_id": task_id, "pdf_key": pdf_key, "success": True, "skipped": True}
        
        try:
            # Process the PDF
            result = processor.process_pdf_from_s3(pdf_key)
            queue.complete_task(task_id, success=result, skipped=False)
            return {"task_id": task_id, "pdf_key": pdf_key, "success": result, "skipped": False}
        
        except Exception as e:
            logger.error(f"Error processing {pdf_key}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            queue.complete_task(task_id, success=False, skipped=False, error=str(e))
            return {"task_id": task_id, "pdf_key": pdf_key, "success": False, "skipped": False, "error": str(e)}
    
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        time.sleep(5)  # Wait before retry
        return None
    except Exception as e:
        logger.error(f"Error in process_pdf_from_queue: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        time.sleep(5)  # Wait before retry
        return None


def process_worker_loop():
    """
    Continuously process PDFs from the queue until interrupted.
    """
    logger.info("Starting PDF processing worker")
    
    # Add a brief delay to ensure Redis is ready
    time.sleep(2)
    
    # Initialize retry count for connection attempts
    connection_attempts = 0
    max_connection_attempts = 5
    
    try:
        while True:
            try:
                result = process_pdf_from_queue()
                
                if not result:
                    # No tasks in queue, wait a bit before checking again
                    time.sleep(5)
                    continue
                
                # Reset connection attempts on successful processing
                connection_attempts = 0
                
                logger.info(f"Processed PDF {result['pdf_key']} with result: {'success' if result['success'] else 'failure'}")
                
            except redis.exceptions.ConnectionError as e:
                connection_attempts += 1
                logger.error(f"Redis connection error (attempt {connection_attempts}/{max_connection_attempts}): {str(e)}")
                
                if connection_attempts >= max_connection_attempts:
                    logger.critical("Maximum Redis connection attempts reached. Exiting worker.")
                    break
                    
                # Exponential backoff for retries
                retry_time = min(30, 2 ** connection_attempts)
                logger.info(f"Retrying in {retry_time} seconds...")
                time.sleep(retry_time)
                
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(5)  # Wait before retrying
            
    except KeyboardInterrupt:
        logger.info("Worker interrupted, shutting down")
    
    logger.info("Worker stopped")


def main():
    """Command-line interface for the Chunkr Processor."""
    parser = argparse.ArgumentParser(description='Process datasheet PDFs with Chunkr and store results in S3.')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Queue command
    queue_parser = subparsers.add_parser('queue', help='Queue PDFs for processing')
    queue_parser.add_argument('--limit', type=int, help='Maximum number of PDFs to queue')
    queue_parser.add_argument('--bucket', type=str, default="chunkr-datasets",
                        help='S3 bucket name (defaults to environment variable S3_BUCKET)')
    queue_parser.add_argument('--dataset', type=str, default="tables-vlm-azure-distill-v1",
                        help='Dataset name/path in the bucket (defaults to environment variable DATASET_NAME)')
    
    # Worker command
    worker_parser = subparsers.add_parser('worker', help='Start a worker to process queued PDFs')
    worker_parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes to start (default: 1)')
    
    # Process command (legacy direct processing)
    process_parser = subparsers.add_parser('process', help='Process PDFs directly (legacy mode)')
    process_parser.add_argument('--limit', type=int, help='Maximum number of PDFs to process')
    process_parser.add_argument('--bucket', type=str, default="chunkr-datasets",
                        help='S3 bucket name (defaults to environment variable S3_BUCKET)')
    process_parser.add_argument('--dataset', type=str, default="tables-vlm-azure-distill-v1",
                        help='Dataset name/path in the bucket (defaults to environment variable DATASET_NAME)')
    process_parser.add_argument('--threads', type=int, default=5,
                        help='Number of parallel processing threads (default: 5)')
    
    args = parser.parse_args()
    
    # Load .env file
    load_dotenv(override=True)
    api_key = os.environ.get("CHUNKR_API_KEY")
    
    if args.command == 'queue':
        print("--- Chunkr Processor: Queuing PDFs ---")
        print(f"Dataset: {args.dataset}")
        print(f"S3 bucket: {args.bucket}")
        
        # Initialize processor to get the raw PDFs list
        processor = ChunkrProcessor(
            s3_bucket=args.bucket,
            dataset_name=args.dataset
        )
        
        # List PDFs in raw-pdfs folder
        pdfs = []
        try:
            paginator = processor.s3_storage.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(
                Bucket=processor.s3_storage.s3_bucket, 
                Prefix=processor.raw_pdfs_prefix
            ):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.pdf'):
                        pdfs.append(obj['Key'])
        except Exception as e:
            logger.error(f"Error listing PDFs in S3: {str(e)}")
            return
        
        # Add PDFs to queue
        from pdf_queue import add_pdfs_to_queue
        count = add_pdfs_to_queue(pdfs, api_key, args.bucket, args.dataset, args.limit)
        
        print(f"Successfully queued {count} PDFs for processing")
    
    elif args.command == 'worker':
        print("--- Chunkr Processor: Starting Worker(s) ---")
        print(f"Starting {args.workers} worker processes")
        
        if args.workers == 1:
            # Single worker mode, just run in this process
            process_worker_loop()
        else:
            # Multi-worker mode, spawn separate processes
            import multiprocessing
            processes = []
            
            try:
                # Start worker processes
                for i in range(args.workers):
                    p = multiprocessing.Process(target=process_worker_loop)
                    p.start()
                    processes.append(p)
                    print(f"Started worker process {i+1}")
                
                # Wait for all processes to complete (or be interrupted)
                for p in processes:
                    p.join()
            
            except KeyboardInterrupt:
                print("Interrupted, stopping workers...")
                for p in processes:
                    p.terminate()
                
                # Wait for processes to terminate
                for p in processes:
                    p.join()
            
            print("All workers stopped")
    
    elif args.command == 'process':
        # Legacy direct processing mode
        print("--- Chunkr Processor: Direct Processing Mode ---")
        print(f"Processing up to {args.limit or 'all'} PDFs from S3 raw-pdfs folder")
        print(f"Dataset: {args.dataset}")
        print(f"S3 bucket: {args.bucket}")
        print(f"Using {args.threads} parallel threads")
        print("WARNING: This mode is not recommended for production use.")
        
        start_time = time.time()
        
        # Initialize the processor with arguments
        processor = ChunkrProcessor(
            s3_bucket=args.bucket,
            dataset_name=args.dataset,
            max_workers=args.threads
        )
        
        # Use the traditional approach (single-process)
        stats = run_single_process(processor, args.limit)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Print summary statistics
        print("\n--- Processing Summary ---")
        print(f"Total time: {elapsed_time:.2f} seconds")
        if "error" in stats:
            print(f"Error: {stats['error']}")
        else:
            print(f"Total PDFs found: {stats.get('total', 'N/A')}")
            print(f"Successfully processed: {stats.get('success', 'N/A')}")
            print(f"Skipped (already processed): {stats.get('skipped', 'N/A')}")
            print(f"Failed: {stats.get('failed', 'N/A')}")
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        processor.cleanup()
        print("Cleanup complete.")
    
    else:
        parser.print_help()


def run_single_process(processor, limit=None):
    """
    Run processing in a single process (no multiprocessing).
    This avoids the pickling issues with multiprocessing.
    
    Args:
        processor: ChunkrProcessor instance
        limit: Optional limit on number of PDFs to process
        
    Returns:
        Dict with processing statistics
    """
    # Ensure output directories exist
    processor._ensure_output_structure()
    
    # List PDFs in raw-pdfs folder
    pdfs = []
    try:
        paginator = processor.s3_storage.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(
            Bucket=processor.s3_storage.s3_bucket, 
            Prefix=processor.raw_pdfs_prefix
        ):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.pdf'):
                    pdfs.append(obj['Key'])
    except Exception as e:
        logger.error(f"Error listing PDFs in S3: {str(e)}")
        return {"error": str(e)}
    
    if limit:
        pdfs = pdfs[:limit]
    
    logger.info(f"Found {len(pdfs)} PDFs to process from S3 raw-pdfs folder")
    
    # Process PDFs
    stats = {
        "total": len(pdfs),
        "success": 0,
        "skipped": 0,
        "failed": 0
    }
    
    for i, pdf_key in enumerate(pdfs):
        processed_count = i + 1
        total_to_process = len(pdfs)
        filename = pdf_key.split('/')[-1]
        base_filename = Path(filename).stem
        
        # Check if outputs already exist
        if processor._check_outputs_exist(base_filename):
            stats["skipped"] += 1
            logger.info(f"({processed_count}/{total_to_process}) Skipped: {base_filename}")
            continue
        
        # Process the PDF
        result = processor.process_pdf_from_s3(pdf_key)
        
        if result:
            stats["success"] += 1
            logger.info(f"({processed_count}/{total_to_process}) Success: {base_filename}")
        else:
            stats["failed"] += 1
            logger.warning(f"({processed_count}/{total_to_process}) Failed: {base_filename}")
    
    return stats


if __name__ == "__main__":
    main()
