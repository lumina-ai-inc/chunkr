import os
import time
import dotenv
from chunkr_ai import Chunkr
from chunkr_ai.models import (
    Configuration, 
    Pipeline,
    SegmentProcessing, 
    GenerationConfig,
    GenerationStrategy,
    SegmentationStrategy,
    LlmProcessing,
    ChunkProcessing
)
import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from typing import IO, Union, Optional, Dict, List, Any
import multiprocessing
import traceback
import img2pdf  # Added for image to PDF conversion
import tempfile  # Added for temporary file handling
import argparse # Added for command-line arguments

# Load environment variables from .env file in the current directory or parent directories
dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

# --- Configuration ---
# S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "chunkr-web")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-west-1")

# Chunkr Configuration
CHUNKR_API_KEY = os.getenv("CHUNKR_API_KEY")

# Path to frontend assets directory
FRONTEND_ASSETS_DIR = os.getenv("FRONTEND_ASSETS_DIR", "frontend_pdfs")

# Supported image extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']

MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))

# --- Image to PDF Conversion Function ---
def convert_image_to_pdf(image_path: str) -> str:
    """
    Convert an image file to PDF format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Path to the generated PDF file (temporary file)
    """
    try:
        # Create a temporary file for the PDF output
        temp_pdf_fd, temp_pdf_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_pdf_fd)  # Close the file descriptor
        
        print(f"Converting image {image_path} to PDF {temp_pdf_path}")
        
        # Convert the image to PDF using img2pdf
        with open(temp_pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_path))
            
        return temp_pdf_path
    except Exception as e:
        print(f"Error converting image to PDF: {e}")
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        raise

# --- Top-level Helper Functions for S3 (used by worker processes) ---
def _static_upload_file_to_s3(s3_client, local_file_path: str, s3_key: str, bucket_name: str, region_name: str) -> str:
    full_s3_key = f"landing_page_v2/{s3_key}"
    print(f"[Worker {os.getpid()}] Uploading {local_file_path} to s3://{bucket_name}/{full_s3_key}...")
    try:
        s3_client.upload_file(local_file_path, bucket_name, full_s3_key)
        s3_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{full_s3_key}"
        print(f"[Worker {os.getpid()}] Successfully uploaded. Public URL: {s3_url}")
        return s3_url
    except FileNotFoundError as e:
        print(f"[Worker {os.getpid()}] ERROR: Local file not found: {local_file_path}")
        raise e
    except ClientError as e:
        print(f"[Worker {os.getpid()}] ERROR uploading file {local_file_path} to S3: {str(e)}")
        raise e

def _static_upload_json_to_s3(s3_client, json_data: Dict[str, Any], s3_key: str, bucket_name: str, region_name: str) -> str:
    full_s3_key = f"landing_page_v2/{s3_key}"
    print(f"[Worker {os.getpid()}] Uploading JSON data to s3://{bucket_name}/{full_s3_key}...")
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=full_s3_key,
            Body=json.dumps(json_data, indent=2),
            ContentType='application/json'
        )
        s3_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{full_s3_key}"
        print(f"[Worker {os.getpid()}] Successfully uploaded JSON. Public URL: {s3_url}")
        return s3_url
    except ClientError as e:
        print(f"[Worker {os.getpid()}] ERROR uploading JSON to S3 for key {s3_key}: {str(e)}")
        raise e

# --- Top-level Worker Function for Multiprocessing ---
def standalone_task_processor(
    local_file_path: str, category_name: str,
    chunkr_api_key_val: str, aws_config: dict,
    global_aws_s3_bucket_name: str, global_aws_region_name: str
) -> Union[Dict[str, Any], Exception]:
    temp_pdf_path = None
    try:
        print(f"[Worker {os.getpid()}] Starting processing for: {category_name} - {os.path.basename(local_file_path)}")
        # Initialize clients within the worker process
        chunkr_client = Chunkr(api_key=chunkr_api_key_val)
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_config["access_key_id"],
            aws_secret_access_key=aws_config["secret_access_key"],
            region_name=global_aws_region_name # Use the global region for the client
        )

        original_basename = os.path.basename(local_file_path)
        _, original_extension = os.path.splitext(original_basename)
        
        file_to_process = local_file_path
        if original_extension.lower() != '.pdf':
            print(f"[Worker {os.getpid()}] File is not a PDF. Converting to PDF first.")
            temp_pdf_path = convert_image_to_pdf(local_file_path)
            file_to_process = temp_pdf_path
            print(f"[Worker {os.getpid()}] Converted image to PDF: {temp_pdf_path}")
        else:
            print(f"[Worker {os.getpid()}] File is already a PDF, using as is.")
        
        new_filename_no_ext = category_name
        new_s3_input_filename = f"{category_name}.pdf"

        print(f"[Worker {os.getpid()}] Will be stored in S3 as: input/{category_name}/{new_s3_input_filename} and output/{category_name}/{new_filename_no_ext}_response.json")

        # --- New Configuration Logic ---
        category_lower = category_name.lower()

        # Global configurations
        llm_pro_config = LlmProcessing(model_id="gpt-4.1", max_completion_tokens=4096)
        global_ignore_headers_and_footers = False # "headers and footers on all" implies False unless overridden

        # Default generation configs
        gen_config_llm_no_extended = GenerationConfig(html=GenerationStrategy.LLM, markdown=GenerationStrategy.LLM)
        gen_config_llm_extended = GenerationConfig(html=GenerationStrategy.LLM, markdown=GenerationStrategy.LLM, extended_context=True)

        # Base configuration values
        final_pipeline = Pipeline.AZURE
        final_high_resolution = True
        final_segmentation_strategy = None # Let Chunkr decide by default
        final_llm_processing = llm_pro_config
        
        # Default segment processing: Picture, Table, Formula to LLM
        default_segment_processing = SegmentProcessing(
            Picture=gen_config_llm_extended,
            Table=gen_config_llm_extended,
            Formula=gen_config_llm_no_extended
        )
        final_segment_processing = default_segment_processing

        # Category-specific overrides
        if category_lower == 'tax':
            final_segmentation_strategy = SegmentationStrategy.PAGE
            # Uses default_segment_processing

        elif category_lower == 'technical': # Separated for specific segment processing
            final_high_resolution = False
            final_segment_processing = SegmentProcessing(
                Picture=gen_config_llm_extended,
                Table=gen_config_llm_extended,
            )
        elif category_lower in ['supply_chain',  'legal']:
            final_high_resolution = False
            # These categories will continue to use default_segment_processing
            # (Picture=extended, Table=extended, Formula=no_extended, Text=default/none)

        elif category_lower in ['medical', 'newspaper']:
            final_pipeline = Pipeline.AZURE
            final_high_resolution = False
            # Uses default_segment_processing

        elif category_lower == 'education': # MODIFIED: Only education now
            final_pipeline = Pipeline.CHUNKR
            # Segment processing for education remains as it was previously for education/misc/miscellaneous
            final_segment_processing = SegmentProcessing(
                Text=gen_config_llm_no_extended,
                Picture=gen_config_llm_extended, 
                Table=gen_config_llm_extended,   
                Formula=gen_config_llm_no_extended,
                Caption=gen_config_llm_no_extended,
                Footnote=gen_config_llm_no_extended,
                ListItem=gen_config_llm_no_extended,
                Page=gen_config_llm_no_extended, 
                PageFooter=gen_config_llm_no_extended,
                PageHeader=gen_config_llm_no_extended,
                SectionHeader=gen_config_llm_no_extended,
                Title=gen_config_llm_no_extended
            )
            # final_high_resolution for education will be the global default (True)

        elif category_lower == 'misc' or category_lower == 'miscellaneous': # NEW block for misc/miscellaneous
            final_pipeline = Pipeline.AZURE
            final_high_resolution = True
            # All specified segment types to use gen_config_llm_no_extended
            final_segment_processing = SegmentProcessing(
                Text=gen_config_llm_no_extended,
                Picture=gen_config_llm_no_extended, # Was gen_config_llm_extended
                Table=gen_config_llm_no_extended,   # Was gen_config_llm_extended
                Formula=gen_config_llm_no_extended,
                Caption=gen_config_llm_no_extended,
                Footnote=gen_config_llm_no_extended,
                ListItem=gen_config_llm_no_extended,
                Page=gen_config_llm_no_extended,
                PageFooter=gen_config_llm_no_extended,
                PageHeader=gen_config_llm_no_extended,
                SectionHeader=gen_config_llm_no_extended,
                Title=gen_config_llm_no_extended
            )
        
        elif category_lower == 'research':
            final_pipeline = Pipeline.AZURE
            final_high_resolution = True
            # Uses default_segment_processing (Picture, Table, Formula with LLM)

        elif category_lower == 'construction':
            # Uses default pipeline (Azure), default high_res (True)
            final_segment_processing = SegmentProcessing(
                Picture=gen_config_llm_extended,
                Table=gen_config_llm_extended,
                Formula=gen_config_llm_extended # Formula gets extended_context here
            )

        elif category_lower in [ 'historical', 'history']:
            # "misc rm" - uses default Azure pipeline, default high_res (True),
            # and default_segment_processing (Picture, Table, Formula with LLM).
            # Text is NOT set to LLM, aligning with "rm".
            pass # Covered by defaults and default_segment_processing

        # For 'consulting' and any other unspecified categories:
        # They will use the defaults:
        # final_pipeline = Pipeline.AZURE
        # final_high_resolution = True
        # final_segmentation_strategy = None
        # final_ignore_headers_and_footers = False
        # final_llm_processing = llm_pro_config
        # final_segment_processing = default_segment_processing

        chunkr_config_obj = Configuration(
            pipeline=final_pipeline,
            segment_processing=final_segment_processing,
            high_resolution=final_high_resolution,
            segmentation_strategy=final_segmentation_strategy,
            llm_processing=final_llm_processing,
            chunk_processing=ChunkProcessing(ignore_headers_and_footers=global_ignore_headers_and_footers)
        )
        
        print(f"[Worker {os.getpid()}] Using Chunkr Config: pipeline={chunkr_config_obj.pipeline}, high_res={chunkr_config_obj.high_resolution}, seg_strategy={chunkr_config_obj.segmentation_strategy}, ignore_hf={chunkr_config_obj.chunk_processing.ignore_headers_and_footers if chunkr_config_obj.chunk_processing else 'None'}, llm_model={chunkr_config_obj.llm_processing.model_id if chunkr_config_obj.llm_processing else 'None'}, SP_keys={list(chunkr_config_obj.segment_processing.model_dump().keys()) if chunkr_config_obj.segment_processing else 'None'} for category: {category_name}")
        # --- End of New Configuration Logic ---

        print(f"[Worker {os.getpid()}] Uploading to Chunkr: {file_to_process} for category {category_name}")
        file_arg: Union[str, Path, IO[bytes]] = file_to_process
        upload_response = chunkr_client.upload(file=file_arg, config=chunkr_config_obj)

        try:
            task_response_data = upload_response.model_dump(mode='json')
        except AttributeError:
            task_response_data = upload_response.dict()

        if not task_response_data:
            error_details = getattr(upload_response, 'error', 'No explicit error details. Task response data was empty.')
            status_attr = getattr(upload_response, 'status', 'Status attribute not found.')
            status_value = status_attr.value if hasattr(status_attr, 'value') else status_attr
            raise Exception(f"Chunkr task for {file_to_process} failed or returned empty. Status: {status_value}. Details: {error_details}")
        print(f"[Worker {os.getpid()}] Chunkr task for {file_to_process} completed.")

        # Use the processed file for S3 upload (either original PDF or converted PDF)
        s3_input_key = f"input/{category_name}/{new_s3_input_filename}"
        s3_input_url = _static_upload_file_to_s3(s3_client, file_to_process, s3_input_key, global_aws_s3_bucket_name, global_aws_region_name)
        if not s3_input_url: # Should be an exception from helper
             raise Exception(f"Failed to upload file {file_to_process} to S3.")

        if 'output' in task_response_data and isinstance(task_response_data['output'], dict):
            task_response_data['output']['pdf_url'] = s3_input_url
        elif 'output' not in task_response_data:
             task_response_data['output'] = {'pdf_url': s3_input_url}
        else: # 'output' exists but is not a dict
            print(f"[Worker {os.getpid()}] Warning: 'output' field in task response for {file_to_process} is not a dictionary. Overwriting with pdf_url.")
            task_response_data['output_meta'] = {'original_output_type': str(type(task_response_data.get('output')))}
            task_response_data['output'] = {'pdf_url': s3_input_url}

        s3_output_key = f"output/{category_name}/{new_filename_no_ext}_response.json"
        s3_output_url = _static_upload_json_to_s3(s3_client, task_response_data, s3_output_key, global_aws_s3_bucket_name, global_aws_region_name)
        if not s3_output_url: # Should be an exception from helper
            print(f"[Worker {os.getpid()}] Failed to upload Chunkr JSON response for {original_basename} to S3. This is unexpected if helper raises.")

        return {
            "category_id": category_name,
            "pdf_name": new_filename_no_ext,
            "original_filename": original_basename,
            "s3_input_url": s3_input_url,
            "s3_output_json_url": s3_output_url,
            "s3_output_json_key": s3_output_key # This is the key relative to "landing_page_v2/"
        }

    except Exception as e:
        return RuntimeError(f"Worker failed for {category_name}/{os.path.basename(local_file_path)}: {type(e).__name__} - {e}\n{traceback.format_exc()}")
    
    finally:
        # Clean up the temporary PDF file if one was created
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
                print(f"[Worker {os.getpid()}] Removed temporary PDF file: {temp_pdf_path}")
            except Exception as e:
                print(f"[Worker {os.getpid()}] Warning: Failed to remove temporary PDF file {temp_pdf_path}: {e}")


class ChunkrS3Uploader:
    def __init__(self):
        if not CHUNKR_API_KEY:
            raise ValueError("Chunkr API key is required. Set CHUNKR_API_KEY in .env file or environment.")
        # self.chunkr removed, initialized in worker

        try:
            # S3 client for _ensure_bucket_exists (called by main process)
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION_NAME # Ensure region is passed for this client too
            )
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Configure them or set in .env file/environment.")
        except Exception as e:
            raise ValueError(f"Error initializing S3 client in Uploader: {e}")

    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=AWS_S3_BUCKET_NAME)
            print(f"Bucket '{AWS_S3_BUCKET_NAME}' already exists in region '{AWS_REGION_NAME}'.")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == '404' or error_code == 'NoSuchBucket':
                print(f"Bucket '{AWS_S3_BUCKET_NAME}' not found. Creating bucket in region '{AWS_REGION_NAME}'...")
                try:
                    if AWS_REGION_NAME == "us-east-1":
                        self.s3_client.create_bucket(Bucket=AWS_S3_BUCKET_NAME)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=AWS_S3_BUCKET_NAME,
                            CreateBucketConfiguration={'LocationConstraint': AWS_REGION_NAME}
                        )
                    waiter = self.s3_client.get_waiter('bucket_exists')
                    waiter.wait(Bucket=AWS_S3_BUCKET_NAME)
                    print(f"Bucket '{AWS_S3_BUCKET_NAME}' created successfully.")
                except ClientError as create_error:
                    raise Exception(f"Could not create bucket '{AWS_S3_BUCKET_NAME}': {create_error}")
            elif error_code == '403':
                 print(f"Access denied for bucket '{AWS_S3_BUCKET_NAME}'. Check permissions.")
                 raise
            else:
                raise Exception(f"Error checking bucket '{AWS_S3_BUCKET_NAME}': {e}")

    def run_with_multiprocessing(self, categories_to_process: Optional[List[str]] = None): # Renamed from run_async
        self._ensure_bucket_exists()
        processed_files_info: List[Dict[str, Any]] = []
        tasks_to_process_params: List[Dict[str,str]] = [] # Stores {"local_file_path": ..., "category_name": ...}

        assets_dir_path = Path(FRONTEND_ASSETS_DIR)
        if not assets_dir_path.is_dir():
            print(f"Error: Assets directory '{assets_dir_path}' not found.")
            return

        all_found_category_dirs = sorted([d.name for d in assets_dir_path.iterdir() if d.is_dir()])

        if categories_to_process:
            # Filter to process only specified categories that exist
            actual_categories_to_run = []
            specified_set = set(categories_to_process)
            found_set = set(all_found_category_dirs)

            for cat_name in all_found_category_dirs: # Keep original order for processing if found
                if cat_name in specified_set:
                    actual_categories_to_run.append(cat_name)
            
            not_found_specified = sorted(list(specified_set.difference(found_set)))
            if not_found_specified:
                print(f"Warning: The following specified categories were not found in '{assets_dir_path}': {not_found_specified}")

            if not actual_categories_to_run:
                print(f"No specified categories matched any existing directories in '{assets_dir_path}'. Aborting.")
                print(f"Specified: {categories_to_process}")
                print(f"Found directories: {all_found_category_dirs}")
                return
            
            category_names = actual_categories_to_run
            print(f"--- Processing ONLY specified categories: {category_names} ---")
        else:
            category_names = all_found_category_dirs
            print(f"--- No specific categories provided, processing ALL found categories in '{assets_dir_path}': {category_names} ---")

        if not category_names:
            print("No categories selected for processing.")
            return
            
        for category_name in category_names:
            category_path = assets_dir_path / category_name
            print(f"\n--- Identifying file for category: {category_name} ---")
            
            found_file_path: Optional[Path] = None
            pdf_files = sorted(list(category_path.glob('*.pdf')))
            if pdf_files:
                found_file_path = pdf_files[0]
            if not found_file_path:
                image_files: List[Path] = []
                for ext in IMAGE_EXTENSIONS:
                    image_files.extend(list(category_path.glob(f'*{ext}')))
                if image_files:
                    found_file_path = sorted(image_files)[0]
            
            if found_file_path:
                print(f"Selected file for category '{category_name}': {found_file_path.name}")
                tasks_to_process_params.append({"local_file_path": str(found_file_path), "category_name": category_name})
            else:
                print(f"No PDF or supported image found in category: {category_name}")
        
        if not tasks_to_process_params:
            print("No files found to process.")
            return

        print(f"\n--- Starting multiprocessing for {len(tasks_to_process_params)} files with max {MAX_CONCURRENT_TASKS} processes ---")
        
        aws_config_dict = {
            "access_key_id": AWS_ACCESS_KEY_ID,
            "secret_access_key": AWS_SECRET_ACCESS_KEY,
            # bucket_name and region_name are passed separately as they are used more directly
            # in URL construction and by S3 client setup.
        }
        
        # Prepare arguments for starmap: list of (local_file_path, category_name, chunkr_key, aws_conf, bucket, region) tuples
        starmap_args = [
            (
                task_param["local_file_path"],
                task_param["category_name"],
                CHUNKR_API_KEY, # Global CHUNKR_API_KEY
                aws_config_dict,
                AWS_S3_BUCKET_NAME, # Global AWS_S3_BUCKET_NAME
                AWS_REGION_NAME     # Global AWS_REGION_NAME
            ) for task_param in tasks_to_process_params
        ]

        results_or_exceptions: List[Union[Dict[str, Any], Exception]] = []
        if starmap_args:
            num_processes = min(MAX_CONCURRENT_TASKS, len(starmap_args))
            with multiprocessing.Pool(processes=num_processes) as pool:
                print(f"Submitting {len(starmap_args)} tasks to process pool with {num_processes} workers...")
                results_or_exceptions = pool.starmap(standalone_task_processor, starmap_args)
                print("All tasks submitted to pool have completed or failed.")
        else:
            print("No tasks to submit to the process pool.")
        
        for i, res_or_exc in enumerate(results_or_exceptions):
            # Get corresponding original task_info for logging context
            original_task_param = tasks_to_process_params[i]
            file_path_for_log = original_task_param["local_file_path"]
            category_name_for_log = original_task_param["category_name"]

            if isinstance(res_or_exc, Exception):
                print(f"  -> FAILED (exception from worker): Category '{category_name_for_log}', File: {Path(file_path_for_log).name}")
                print(f"     Error: {res_or_exc}") # Exception includes traceback string now
            elif isinstance(res_or_exc, dict): # Successful result
                processed_files_info.append(res_or_exc)
                print(f"  -> COMPLETED: Category '{res_or_exc['category_id']}', File: {Path(file_path_for_log).name}")
                print(f"    -> For Home.tsx: category_id='{res_or_exc['category_id']}', pdfName='{res_or_exc['pdf_name']}'") 
            else: # Should not happen if worker returns dict or Exception
                print(f"  -> FAILED (unexpected return type from worker: {type(res_or_exc)}): Category '{category_name_for_log}', File: {Path(file_path_for_log).name}")
        
        processed_files_info.sort(key=lambda x: x["category_id"])

        print("\n--- Script Finished ---")
        if processed_files_info:
            print("\nSuccessfully processed and uploaded:")
            for info in processed_files_info:
                print(f"  Category: {info['category_id']}, File: {info['original_filename']}")
                print(f"    S3 Input: {info['s3_input_url']}")
                print(f"    S3 Output JSON: {info.get('s3_output_json_url', 'Upload Failed or Not Applicable')}")
            
            print("\n--- Configuration for Home.tsx ---")
            self.generate_tsx_config(processed_files_info)
        else:
            print("No files were processed successfully.")

    def generate_tsx_config(self, processed_files_info: List[Dict[str, str]]):
        tsx_categories = []
        for info in processed_files_info:
            label = info['category_id'].replace('_', ' ').replace('-', ' ').title()
            if "Pdfs" in label: # "frontend_pdfs" -> "Frontend"
                label = label.replace("Pdfs", "").strip()
            elif "Pdf" in label:
                label = label.replace("Pdf", "").strip()


            tsx_categories.append({
                "id": info['category_id'],
                "label": f"{label}", 
                "pdfName": info['pdf_name'] # This now correctly uses category_name from the result
            })
        
        print("\nconst DOCUMENT_CATEGORIES: DocumentCategory[] = [")
        for i, cat in enumerate(tsx_categories):
            comma = "," if i < len(tsx_categories) - 1 else ""
            print(f"  {{ id: \"{cat['id']}\", label: \"{cat['label']}\", pdfName: \"{cat['pdfName']}\" }}{comma}")
        print("];")
        
        base_s3_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_REGION_NAME}.amazonaws.com/landing_page_v2" # Added landing_page_v2
        print(f"\nconst BASE_URL = \"{base_s3_url}\";")
        print("\n// Remember to update Home.tsx with these values and adjust fetchTaskResponse if needed.")
        print("// Ensure selectedCategory in Home.tsx is initialized to one of the new category IDs, e.g., DOCUMENT_CATEGORIES[0]?.id")


if __name__ == "__main__":
    print("Starting Chunkr S3 Uploader script (Multiprocessing Version)...")
    print(f"Reading configuration from .env and environment variables.")

    parser = argparse.ArgumentParser(description="Upload PDFs/images to Chunkr and S3, with category-specific configurations.")
    parser.add_argument(
        '--categories', 
        nargs='*', 
        help='A list of category names (directory names in FRONTEND_ASSETS_DIR) to process. If not provided, all categories will be processed.'
    )
    args = parser.parse_args()

    # print(f"CHUNKR_API_URL: {CHUNKR_API_URL}") # User removed CHUNKR_API_URL variable
    print(f"AWS_S3_BUCKET_NAME: {AWS_S3_BUCKET_NAME}")
    print(f"AWS_REGION_NAME: {AWS_REGION_NAME}")
    print(f"FRONTEND_ASSETS_DIR: {FRONTEND_ASSETS_DIR}")
    print(f"MAX_CONCURRENT_TASKS (processes): {MAX_CONCURRENT_TASKS}")
    
    uploader = ChunkrS3Uploader()
    uploader.run_with_multiprocessing(categories_to_process=args.categories) # Changed from asyncio.run
