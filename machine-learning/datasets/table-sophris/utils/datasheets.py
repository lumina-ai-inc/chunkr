import os
import csv
import requests
import logging
import hashlib
import tempfile
import time
import concurrent.futures
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from storage import TableS3Storage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('datasheets')

class DatasheetDownloader:
    """
    Downloads and stores datasheets (PDFs) from URLs listed in a CSV file.
    Handles duplicate checking and S3 storage integration.
    """

    def __init__(self, s3_bucket=None, dataset_name=None, max_retries=3, timeout=120):
        """
        Initialize the datasheet downloader.

        Args:
            s3_bucket: S3 bucket name (if None, uses environment variable S3_BUCKET)
            dataset_name: Dataset name (if None, uses environment variable DATASET_NAME)
            max_retries: Maximum number of download retry attempts
            timeout: Download timeout in seconds
        """
        self.s3_storage = TableS3Storage(s3_bucket, dataset_name)
        self.max_retries = max_retries
        self.timeout = timeout
        self.local_temp_dir = Path("temp_datasheet_downloads")
        self.local_temp_dir.mkdir(exist_ok=True)
        
        # Define the path for raw PDFs in S3
        self.raw_pdf_prefix = f"{self.s3_storage.base_prefix}{self.s3_storage.dataset_name}/raw-pdfs/"

    def _ensure_bucket_exists(self):
        """
        Ensure the S3 bucket exists, creating it if necessary.
        
        Returns:
            bool: Success or failure
        """
        if not self.s3_storage.is_enabled():
            return False
            
        try:
            # Check if bucket exists
            try:
                self.s3_storage.s3_client.head_bucket(Bucket=self.s3_storage.s3_bucket)
                logger.info(f"S3 bucket exists: {self.s3_storage.s3_bucket}")
                return True
            except Exception as e:
                # If error is not 404 (not found), it could be permissions or other issue
                if hasattr(e, 'response') and e.response.get('Error', {}).get('Code') != '404':
                    logger.error(f"Error accessing bucket: {str(e)}")
                    return False
                
                # Bucket doesn't exist, create it
                logger.info(f"Creating S3 bucket: {self.s3_storage.s3_bucket}")
                
                # Get the current AWS region
                region = self.s3_storage.s3_client.meta.region_name
                
                # Create bucket with regional constraint if not in us-east-1
                if region == 'us-east-1':
                    self.s3_storage.s3_client.create_bucket(
                        Bucket=self.s3_storage.s3_bucket
                    )
                else:
                    self.s3_storage.s3_client.create_bucket(
                        Bucket=self.s3_storage.s3_bucket,
                        CreateBucketConfiguration={
                            'LocationConstraint': region
                        }
                    )
                
                logger.info(f"Successfully created bucket: {self.s3_storage.s3_bucket}")
                return True
                
        except Exception as e:
            logger.error(f"Error ensuring bucket exists: {str(e)}")
            return False

    def _ensure_pdf_structure(self):
        """Ensure the raw-pdfs folder structure exists in S3."""
        if not self.s3_storage.is_enabled():
            return False
            
        # First ensure the bucket exists
        if not self._ensure_bucket_exists():
            return False
            
        try:
            self.s3_storage.s3_client.put_object(
                Bucket=self.s3_storage.s3_bucket,
                Key=self.raw_pdf_prefix,
                Body=''
            )
            logger.info(f"Ensured raw-pdfs folder structure in S3")
            return True
        except Exception as e:
            logger.error(f"Error ensuring raw-pdfs structure: {str(e)}")
            return False

    def _parse_csv(self, csv_file_path: str) -> List[Dict]:
        """
        Parse the CSV file containing datasheet URLs.
        
        Args:
            csv_file_path: Path to the CSV file
            
        Returns:
            List of dictionaries containing datasheet information
        """
        datasheets = []
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Only add rows that have a valid URL
                    if ('DatasheetURL' in row and 
                        row['DatasheetURL'].strip() and 
                        row['DatasheetURL'].startswith('http')):
                        datasheets.append(row)
                    else:
                        logger.warning(f"Skipping row with invalid URL: {row.get('DatasheetURL', 'NO_URL')}")
            
            logger.info(f"Parsed {len(datasheets)} valid datasheet entries from {csv_file_path}")
            return datasheets
        except Exception as e:
            logger.error(f"Error parsing CSV file {csv_file_path}: {str(e)}")
            return []

    def _generate_filename(self, datasheet: Dict) -> str:
        """
        Generate a unique filename for the datasheet PDF.
        
        Args:
            datasheet: Dictionary containing datasheet information
            
        Returns:
            A filename for the PDF
        """
        # Create a base name using manufacturer and part number if available
        parts = []
        
        if 'Manufacturer' in datasheet and datasheet['Manufacturer']:
            parts.append(datasheet['Manufacturer'].replace(' ', '_'))
            
        if 'ManufacturerPartNumber' in datasheet and datasheet['ManufacturerPartNumber']:
            parts.append(datasheet['ManufacturerPartNumber'].replace(' ', '_'))
        
        # If we don't have enough info, use the URL's filename
        if not parts:
            url_path = urlparse(datasheet['DatasheetURL']).path
            url_filename = os.path.basename(url_path)
            if url_filename and '.' in url_filename:
                parts = [url_filename.rsplit('.', 1)[0]]
        
        # If we still don't have a name, create a hash of the URL
        if not parts:
            parts = [hashlib.md5(datasheet['DatasheetURL'].encode()).hexdigest()[:12]]
        
        # Combine parts and ensure .pdf extension
        filename = '-'.join(parts)
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
            
        return filename

    def _download_pdf(self, url: str, output_path: Path) -> bool:
        """
        Download a PDF from a URL to the specified path with faster failure handling.
        
        Args:
            url: URL of the PDF
            output_path: Path where the PDF should be saved
            
        Returns:
            bool: Success or failure
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Downloading {url} (attempt {attempt}/{self.max_retries})")
                
                # Use a shorter timeout for the initial connection
                with requests.get(
                    url, 
                    stream=True, 
                    timeout=(10, self.timeout),  # (connect timeout, read timeout)
                    headers=headers,
                    allow_redirects=True
                ) as response:
                    response.raise_for_status()
                    
                    # Quick check of content type
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                        logger.warning(f"Skipping: URL may not be a PDF: {url}")
                        return False
                    
                    # Quick check of content length
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > 100 * 1024 * 1024:  # > 100MB
                        logger.warning(f"Skipping: File too large ({int(content_length)/1024/1024:.1f}MB): {url}")
                        return False
                    
                    with open(output_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    
                    # Quick PDF validation
                    if output_path.stat().st_size > 0:
                        with open(output_path, 'rb') as f:
                            magic_number = f.read(4)
                            if magic_number == b'%PDF':
                                logger.info(f"Successfully downloaded {url}")
                                return True
                            else:
                                logger.warning(f"Skipping: Not a valid PDF from {url}")
                                return False
                    
                    logger.warning(f"Skipping: Empty file from {url}")
                    return False
                    
            except requests.RequestException as e:
                logger.error(f"Error downloading {url} (attempt {attempt}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries:
                    # Shorter wait times between retries
                    wait_time = 5 * attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Skipping after {self.max_retries} failed attempts: {url}")
                    return False
                    
        return False

    def _check_duplicate_in_s3(self, filename: str) -> bool:
        """
        Check if a datasheet with the same filename already exists in S3.
        
        Args:
            filename: Filename to check
            
        Returns:
            bool: True if duplicate exists, False otherwise
        """
        if not self.s3_storage.is_enabled():
            return False
            
        s3_key = f"{self.raw_pdf_prefix}{filename}"
        
        try:
            self.s3_storage.s3_client.head_object(
                Bucket=self.s3_storage.s3_bucket,
                Key=s3_key
            )
            # If we get here, the file exists
            logger.info(f"Datasheet already exists in S3: {s3_key}")
            return True
        except Exception:
            # File doesn't exist
            return False

    def _upload_pdf_to_s3(self, local_path: Path, filename: str) -> bool:
        """
        Upload a PDF to S3.
        
        Args:
            local_path: Local path to the PDF file
            filename: Filename to use in S3
            
        Returns:
            bool: Success or failure
        """
        if not self.s3_storage.is_enabled():
            return False
            
        s3_key = f"{self.raw_pdf_prefix}{filename}"
        
        try:
            with open(local_path, 'rb') as file:
                self.s3_storage.s3_client.upload_fileobj(
                    file,
                    self.s3_storage.s3_bucket,
                    s3_key,
                    ExtraArgs={
                        'ContentType': 'application/pdf'
                    },
                    Config=self.s3_storage.transfer_config
                )
            logger.info(f"Uploaded datasheet to S3: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading datasheet to S3: {str(e)}")
            return False

    def _process_single_datasheet(self, datasheet: Dict) -> Tuple[bool, str]:
        """
        Process a single datasheet - download and upload to S3.
        
        Args:
            datasheet: Dictionary containing datasheet information
            
        Returns:
            Tuple[bool, str]: (success, status_message)
        """
        # Generate a filename for the PDF
        filename = self._generate_filename(datasheet)
        
        # Check if the datasheet already exists in S3
        if self._check_duplicate_in_s3(filename):
            return (False, f"Skipping (already exists): {filename}")
        
        # Create a temporary path for the download
        temp_path = self.local_temp_dir / filename
        
        # Download the PDF
        if self._download_pdf(datasheet['DatasheetURL'], temp_path):
            # Upload to S3
            if self._upload_pdf_to_s3(temp_path, filename):
                # Remove the temporary file
                temp_path.unlink(missing_ok=True)
                return (True, f"Successfully downloaded and uploaded: {filename}")
            else:
                # Remove the temporary file
                temp_path.unlink(missing_ok=True)
                return (False, f"Failed to upload: {filename}")
        else:
            return (False, f"Failed to download: {filename}")

    def process_csv_multithreaded(self, csv_file_path: str, max_workers: int = 8) -> Tuple[int, int, int]:
        """
        Process a CSV file with datasheet URLs using multiple threads.
        
        Args:
            csv_file_path: Path to the CSV file
            max_workers: Maximum number of worker threads
            
        Returns:
            Tuple[int, int, int]: (successfully processed, duplicates, failures)
        """
        # Ensure the bucket and S3 structure exist
        if not self._ensure_bucket_exists():
            logger.error("Failed to ensure S3 bucket exists, cannot proceed with processing")
            return 0, 0, 0
        
        # Ensure the S3 structure exists
        if not self._ensure_pdf_structure():
            logger.error("Failed to ensure S3 directory structure, cannot proceed with processing")
            return 0, 0, 0
        
        # Parse the CSV file
        datasheets = self._parse_csv(csv_file_path)
        
        if not datasheets:
            logger.warning("No datasheets found in CSV file")
            return 0, 0, 0
        
        # Track statistics
        success_count = 0
        duplicate_count = 0
        failure_count = 0
        total_count = len(datasheets)
        processed_count = 0
        
        logger.info(f"Processing {total_count} datasheets with {max_workers} worker threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_datasheet = {executor.submit(self._process_single_datasheet, datasheet): datasheet 
                                 for datasheet in datasheets}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_datasheet):
                processed_count += 1
                success, message = future.result()
                
                if success:
                    success_count += 1
                elif "already exists" in message:
                    duplicate_count += 1
                else:
                    failure_count += 1
                
                # Log progress every 10 files or when hitting certain percentages
                if processed_count % 10 == 0 or processed_count in [total_count//4, total_count//2, total_count*3//4]:
                    progress = (processed_count / total_count) * 100
                    logger.info(f"Progress: {processed_count}/{total_count} ({progress:.1f}%) - "
                              f"Success: {success_count}, Duplicates: {duplicate_count}, "
                              f"Failures: {failure_count}")
                else:
                    logger.info(message)
        
        # Summary
        logger.info(f"Processing complete: {success_count} downloaded, {duplicate_count} duplicates, "
                    f"{failure_count} failures out of {total_count} total")
        return success_count, duplicate_count, failure_count

    def list_downloaded_datasheets(self) -> List[str]:
        """
        List all datasheets that have been downloaded to S3.
        
        Returns:
            List of datasheet filenames
        """
        if not self.s3_storage.is_enabled():
            return []

        datasheets = []
        paginator = self.s3_storage.s3_client.get_paginator('list_objects_v2')
        
        try:
            for page in paginator.paginate(Bucket=self.s3_storage.s3_bucket, Prefix=self.raw_pdf_prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.pdf'):
                        filename = key.split('/')[-1]
                        datasheets.append(filename)
            
            logger.info(f"Found {len(datasheets)} datasheets in S3")
            return sorted(datasheets)
        except Exception as e:
            logger.error(f"Error listing datasheets: {str(e)}")
            return []

    def download_datasheet_from_s3(self, filename: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Download a datasheet from S3 to a local directory.
        
        Args:
            filename: Name of the datasheet file
            output_dir: Directory to save the file (defaults to temp directory)
            
        Returns:
            Path to the downloaded file or None if download failed
        """
        if not self.s3_storage.is_enabled():
            return None
            
        if output_dir is None:
            output_dir = self.local_temp_dir
        
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / filename
        
        s3_key = f"{self.raw_pdf_prefix}{filename}"
        
        try:
            self.s3_storage.s3_client.download_file(
                self.s3_storage.s3_bucket,
                s3_key,
                str(output_path),
                Config=self.s3_storage.transfer_config
            )
            logger.info(f"Downloaded {filename} from S3 to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error downloading {filename} from S3: {str(e)}")
            return None

    def cleanup(self):
        """Clean up temporary downloaded files."""
        try:
            for item in self.local_temp_dir.iterdir():
                if item.is_file():
                    item.unlink()
            logger.info(f"Cleaned up temporary directory: {self.local_temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")


def main():
    """Test downloading PDFs from the datasheet URLs CSV with multithreading support."""
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download datasheets from URLs listed in a CSV file.')
    parser.add_argument('--csv', type=str, default="data/datasheet_urls.csv",
                      help='Path to the CSV file containing datasheet URLs')
    parser.add_argument('--limit', type=int, default=50000,
                      help='Maximum number of PDFs to download (default: 100)')
    parser.add_argument('--bucket', type=str, default=None,
                      help='S3 bucket name (defaults to environment variable)')
    parser.add_argument('--dataset', type=str, default="tables-vlm-azure-distill-v1",
                      help='Dataset name/path in the bucket')
    parser.add_argument('--threads', type=int, default=8,
                      help='Number of parallel download threads (default: 4)')
    parser.add_argument('--timeout', type=int, default=120,
                      help='Timeout in seconds for each download (default: 120)')
    parser.add_argument('--retries', type=int, default=3,
                      help='Number of retry attempts for each download (default: 3)')
    args = parser.parse_args()
    
    print(f"Starting test: Downloading up to {args.limit} PDFs from {args.csv}")
    print(f"Dataset path: {args.dataset}")
    print(f"S3 bucket: {args.bucket or 'From environment variable'}")
    print(f"Using {args.threads} parallel threads")
    print(f"Download timeout: {args.timeout} seconds, retries: {args.retries}")
    
    # Set up the downloader
    start_time = time.time()
    downloader = DatasheetDownloader(
        s3_bucket=args.bucket, 
        dataset_name=args.dataset,
        max_retries=args.retries,
        timeout=args.timeout
    )
    
    csv_file_path = args.csv
    
    if not Path(csv_file_path).exists():
        print(f"CSV file not found: {csv_file_path}")
        return
    
    # Parse the CSV file but limit to the specified number
    datasheets = downloader._parse_csv(csv_file_path)[:args.limit]
    print(f"Found {len(datasheets)} datasheet URLs in CSV (limited to {args.limit})")
    
    # Process using multithreading
    if args.threads <= 1:
        # Single-threaded mode
        success_count = 0
        duplicate_count = 0
        failure_count = 0
        
        # Ensure bucket and S3 structure
        if not downloader._ensure_bucket_exists():
            print("Failed to ensure S3 bucket exists, cannot proceed with test")
            return
        
        if not downloader._ensure_pdf_structure():
            print("Failed to ensure S3 directory structure, cannot proceed with test")
            return
        
        # Process each datasheet
        for i, datasheet in enumerate(datasheets, 1):
            print(f"Processing datasheet {i}/{len(datasheets)}")
            
            success, message = downloader._process_single_datasheet(datasheet)
            print(f"  - {message}")
            
            if success:
                success_count += 1
            elif "already exists" in message:
                duplicate_count += 1
            else:
                failure_count += 1
    else:
        # Multi-threaded mode
        success_count, duplicate_count, failure_count = downloader.process_csv_multithreaded(
            csv_file_path=csv_file_path, 
            max_workers=args.threads
        )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Summary
    print("\n--- Test Results ---")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Successfully downloaded and uploaded: {success_count}")
    print(f"Duplicates skipped: {duplicate_count}")
    print(f"Failures: {failure_count}")
    
    # List all datasheets in S3
    all_datasheets = downloader.list_downloaded_datasheets()
    print(f"Total datasheets now in S3: {len(all_datasheets)}")
    
    # Clean up
    downloader.cleanup()
    print("Cleaned up temporary files")

if __name__ == "__main__":
    main()
