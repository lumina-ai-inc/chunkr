import os
import psycopg
from psycopg.rows import dict_row, DictRow
from dotenv import load_dotenv
from nomic import AtlasDataset
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Union
from psycopg import sql
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
from schema import SegmentType, DefectType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Assumes the script is in machine-learning/datasets/dla/src/dla/
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent.parent / '.env'
TAGGING_SET_TABLE_NAME = "dla_selected_tagging_pages"
PAGE_PROFILES_TABLE_NAME = "page_profiles" # Make sure this is your actual page profiles table name
NOMIC_DATASET_NAME = "dla_tagging_set_images_v2" # Choose a unique name for your Nomic dataset
# IMAGE_BASE_PATH is no longer needed as we fetch from S3

# AWS S3 Configuration (names from dla_explorer.py/.env conventions)
# These will be loaded from .env by load_environment()
AWS_ACCESS_KEY_ID_ENV_VAR = "AWS__ACCESS_KEY"
AWS_SECRET_ACCESS_KEY_ENV_VAR = "AWS__SECRET_KEY"
AWS_S3_BUCKET_NAME_ENV_VAR = "AWS_BUCKET_NAME"
AWS_S3_ENDPOINT_URL_ENV_VAR = "AWS__ENDPOINT"
AWS_REGION_NAME_ENV_VAR = "AWS_REGION_NAME"

# --- Helper Functions ---

def load_environment(env_path: Path) -> None:
    """Load environment variables from .env file."""
    if env_path.is_file():
        load_dotenv(env_path, override=True)
        logging.info(f".env file loaded from {env_path}")
    else:
        logging.warning(f"Warning: .env file not found at {env_path}")

def connect_postgres(db_url: str) -> Optional[psycopg.Connection]:
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg.connect(db_url, row_factory=dict_row) # type: ignore
        logging.info("Successfully connected to PostgreSQL database.")
        return conn
    except psycopg.Error as e:
        logging.error(f"Failed to connect to PostgreSQL database: {e}")
        return None

def get_s3_client() -> Optional[Any]:
    """Initialize and return an S3 client."""
    aws_access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_VAR)
    aws_secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_VAR)
    aws_s3_bucket_name = os.getenv(AWS_S3_BUCKET_NAME_ENV_VAR)
    aws_s3_endpoint_url = os.getenv(AWS_S3_ENDPOINT_URL_ENV_VAR)
    aws_region_name = os.getenv(AWS_REGION_NAME_ENV_VAR)

    if not all([aws_access_key_id, aws_secret_access_key, aws_s3_bucket_name]):
        logging.error(
            "AWS S3 credentials or bucket name not fully configured. "
            f"Please ensure {AWS_ACCESS_KEY_ID_ENV_VAR}, {AWS_SECRET_ACCESS_KEY_ENV_VAR}, and "
            f"{AWS_S3_BUCKET_NAME_ENV_VAR} environment variables are set."
        )
        return None
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
            endpoint_url=aws_s3_endpoint_url
        )
        logging.info("S3 client created successfully.")
        return client
    except ClientError as e:
        logging.error(f"Error creating S3 client: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error creating S3 client: {e}")
        return None

def fetch_s3_image_bytes(s3_client: Any, s3_bucket_name: str, image_key: str) -> Optional[BytesIO]:
    """Fetches an image from S3 and returns its content as BytesIO."""
    if not image_key:
        logging.warning("No S3 image key provided.")
        return None
    try:
        response = s3_client.get_object(Bucket=s3_bucket_name, Key=image_key)
        image_content = response['Body'].read()
        return BytesIO(image_content)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            logging.warning(f"S3 Error: Image key not found - s3://{s3_bucket_name}/{image_key}")
        else:
            logging.error(f"S3 ClientError downloading {image_key}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected Error downloading image {image_key}: {e}")
        return None

def prepare_metadata(profile_data: DictRow, segments: List[str], defects: List[str]) -> Dict[str, Any]:
    """
    Prepares a single page's profile data for Nomic Atlas.
    Converts lists to comma-separated strings and ensures basic types.
    """
    metadata: Dict[str, Union[str, int, float, bool, None]] = {}
    metadata['page_identifier'] = str(profile_data['page_identifier'])
    # aws_s3_image_key is part of profile_data but not directly sent to Nomic as metadata unless desired.
    # It's used to fetch the image.

    direct_copy_fields = [
        'doc_id', 'page_domain', 'page_domain_category', 'page_source',
        'page_original_complexity', 'page_layout_style',
        'page_effective_complexity_category', 'page_effective_complexity_numeric',
        'page_primary_language'
    ]
    numeric_fields = [
        'page_primary_source_numeric', 'page_is_non_english_numeric',
        'page_distinct_segment_types_count', 'page_has_table_segment_numeric',
        'page_has_form_segment_numeric', 'page_has_picture_segment_numeric',
        'page_has_defects_numeric', 'page_distinct_defect_types_count'
    ]

    for field_key in direct_copy_fields:
        value = profile_data.get(field_key)
        if value is not None:
            metadata[field_key] = str(value)

    for field_key in numeric_fields:
        value = profile_data.get(field_key)
        if value is not None:
            if isinstance(value, bool):
                metadata[field_key] = int(value)
            elif isinstance(value, (int, float)):
                metadata[field_key] = value
            else:
                try:
                    metadata[field_key] = int(value)
                except (ValueError, TypeError):
                    logging.warning(f"Could not convert field '{field_key}' value '{value}' to int for page {metadata['page_identifier']}. Storing as string.")
                    metadata[field_key] = str(value)

    # Add combined segment and defect strings
    metadata['page_segments_str'] = ",".join(segments) if segments else ""
    metadata['page_defects_str'] = ",".join(defects) if defects else ""

    # One-hot encode each segment and defect type
    for seg_type in SegmentType:
        metadata[f"segment_{seg_type.name}"] = int(seg_type.value in segments)
    for defect_type in DefectType:
        metadata[f"defect_{defect_type.name}"] = int(defect_type.value in defects)

    return metadata

# --- Main Execution ---

def main():
    load_environment(DEFAULT_ENV_PATH)

    source_db_url = os.getenv("SOURCE_DATABASE_URL")
    aws_s3_bucket_name = os.getenv(AWS_S3_BUCKET_NAME_ENV_VAR)

    if not source_db_url:
        logging.error("Missing SOURCE_DATABASE_URL environment variable.")
        return
    if not aws_s3_bucket_name:
        logging.error(f"Missing {AWS_S3_BUCKET_NAME_ENV_VAR} environment variable for S3 bucket name.")
        return

    pg_conn = connect_postgres(source_db_url)
    s3_client = get_s3_client()

    if not pg_conn or not s3_client:
        logging.error("Failed to connect to PostgreSQL or S3. Please check configurations and logs.")
        return

    all_metadata_for_nomic: List[Dict[str, Any]] = []
    all_image_blobs: List[BytesIO] = [] # Changed from image_paths to image_blobs
    page_identifiers_to_upload: List[str] = []

    try:
        with pg_conn.cursor() as cur:
            cur.execute(f"SELECT page_identifier FROM {TAGGING_SET_TABLE_NAME};")
            selected_pages_rows: List[DictRow] = cur.fetchall() # type: ignore
            if not selected_pages_rows:
                logging.info(f"No pages found in {TAGGING_SET_TABLE_NAME}. Exiting.")
                return
            
            page_identifiers_to_upload = [str(row['page_identifier']) for row in selected_pages_rows]
            logging.info(f"Fetched {len(page_identifiers_to_upload)} page identifiers for upload.")

            # Fetch page profiles INCLUDING aws_s3_image_key
            query = sql.SQL("SELECT *, aws_s3_image_key FROM {} WHERE page_identifier = ANY(%s);").format(
                sql.Identifier(PAGE_PROFILES_TABLE_NAME)
            )
            cur.execute(query, (page_identifiers_to_upload,))
            page_profiles_list: List[DictRow] = cur.fetchall() # type: ignore
            logging.info(f"Fetched {len(page_profiles_list)} page profiles.")
            
            profiles_dict: Dict[str, DictRow] = {str(profile['page_identifier']): profile for profile in page_profiles_list}
            # Fetch segments and defects from junction tables
            cur.execute(
                sql.SQL(
                    "SELECT page_identifier, array_agg(segment_type) AS segments "
                    "FROM page_segments WHERE page_identifier = ANY(%s) GROUP BY page_identifier;"
                ),
                (page_identifiers_to_upload,)
            )
            segments_rows: List[DictRow] = cur.fetchall()  # type: ignore
            segments_map = {str(row['page_identifier']): row['segments'] or [] for row in segments_rows}  # type: ignore

            cur.execute(
                sql.SQL(
                    "SELECT page_identifier, array_agg(defect_type) AS defects "
                    "FROM page_defects WHERE page_identifier = ANY(%s) GROUP BY page_identifier;"
                ),
                (page_identifiers_to_upload,)
            )
            defects_rows: List[DictRow] = cur.fetchall()  # type: ignore
            defects_map = {str(row['page_identifier']): row['defects'] or [] for row in defects_rows}  # type: ignore

        # 3 & 4. Concurrent download & upload pipeline
        from concurrent.futures import ThreadPoolExecutor
        from threading import Thread
        from queue import Queue

        num_workers = int(os.getenv('NUM_WORKERS', '8'))
        batch_size = 500
        task_queue: Queue = Queue(maxsize=num_workers * 2)

        # Initialize Nomic Atlas dataset
        logging.info(f"Initializing Nomic Atlas dataset: {NOMIC_DATASET_NAME}")
        dataset = AtlasDataset(
            identifier=NOMIC_DATASET_NAME,
            unique_id_field='page_identifier',
            is_public=False
        )
        logging.info(f"Dataset '{dataset.identifier}' initialized/loaded.")

        def uploader():
            blobs_buffer = []
            meta_buffer = []
            while True:
                item = task_queue.get()
                if item is None:  # Sentinel
                    break
                blob_bytes, meta = item
                blobs_buffer.append(blob_bytes)
                meta_buffer.append(meta)
                if len(blobs_buffer) >= batch_size:
                    dataset.add_data(blobs=blobs_buffer, data=meta_buffer)  # type: ignore
                    logging.info(f"Uploaded batch of {len(blobs_buffer)} items")
                    blobs_buffer.clear()
                    meta_buffer.clear()
                task_queue.task_done()
            # Flush remaining
            if blobs_buffer:
                dataset.add_data(blobs=blobs_buffer, data=meta_buffer)  # type: ignore
                logging.info(f"Uploaded final batch of {len(blobs_buffer)} items")
            task_queue.task_done()

        uploader_thread = Thread(target=uploader, daemon=True)
        uploader_thread.start()

        def process_page(page_id: str):
            profile_data = profiles_dict.get(page_id)
            if not profile_data:
                logging.warning(f"No profile for page {page_id}, skipping.")
                return
            key = profile_data.get('aws_s3_image_key')
            if not key:
                logging.warning(f"No 'aws_s3_image_key' for page {page_id}, skipping.")
                return
            image_io = fetch_s3_image_bytes(s3_client, aws_s3_bucket_name, str(key))
            if not image_io:
                logging.warning(f"Failed to fetch image for page {page_id}, skipping.")
                return
            blob_bytes = image_io.getvalue()
            meta = prepare_metadata(profile_data, segments_map.get(page_id, []), defects_map.get(page_id, []))
            task_queue.put((blob_bytes, meta))

        total_pages = len(page_identifiers_to_upload)
        logging.info(f"Starting download/upload pipeline for {total_pages} pages with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(process_page, page_identifiers_to_upload))

        # Wait for processing and uploading to finish
        task_queue.join()
        task_queue.put(None)
        uploader_thread.join()

        # Create index
        logging.info("Creating Nomic Atlas map (index)... This may take some time.")
        atlas_map = dataset.create_index()
        logging.info("Nomic Atlas map creation initiated.")
        if atlas_map and hasattr(atlas_map, 'map_link'):
            logging.info(f"Map successfully updated: {atlas_map.map_link}")
        else:
            logging.info("Check the Nomic Atlas dashboard for progress.")
    except Exception as e:
        logging.error(f"An error occurred during the pipeline: {e}")

    finally:
        if pg_conn and not pg_conn.closed:
            pg_conn.close()
            logging.info("PostgreSQL connection closed.")

if __name__ == "__main__":
    main() 