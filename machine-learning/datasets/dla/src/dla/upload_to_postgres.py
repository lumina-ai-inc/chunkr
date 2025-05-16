# Filename: upload_to_postgres.py
import os
import json
import time
import argparse
import boto3
import psycopg # Using psycopg 3+ (recommended) or psycopg2
import subprocess # Added for running pg_dump
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple

# --- Configuration Loading ---


load_dotenv(override=True)


# --- AWS Configuration ---
AWS_ACCESS_KEY = os.environ.get("AWS__ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS__SECRET_KEY")
AWS_ENDPOINT = os.environ.get("AWS__ENDPOINT") # Optional
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
# Base prefix where domain folders reside
AWS_S3_BASE_PREFIX = "dla-dataset/"

# --- PostgreSQL Configuration ---
POSTGRES_DB_NAME = os.environ.get("POSTGRES_DB_NAME")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_HOST = os.environ.get("DATABASE_URL", "").split('@')[-1].split(':')[0] if '@' in os.environ.get("DATABASE_URL", "") else None # Extract host from URL if possible
POSTGRES_PORT = os.environ.get("DATABASE_URL", "").split(':')[-1].split('/')[0] if ':' in os.environ.get("DATABASE_URL", "") else "5432" # Extract port

# --- Backup Configuration ---
POSTGRES_BACKUP_DIR = os.environ.get("POSTGRES_BACKUP_DIR", "db_backups") # Default backup directory

# --- Validation ---
missing_vars = []
if not AWS_ACCESS_KEY: missing_vars.append("AWS__ACCESS_KEY")
if not AWS_SECRET_KEY: missing_vars.append("AWS__SECRET_KEY")
if not AWS_BUCKET_NAME: missing_vars.append("AWS_BUCKET_NAME")
if not POSTGRES_DB_NAME: missing_vars.append("POSTGRES_DB_NAME")
if not POSTGRES_USER: missing_vars.append("POSTGRES_USER")
if not POSTGRES_PASSWORD: missing_vars.append("POSTGRES_PASSWORD")
if not POSTGRES_HOST: missing_vars.append("DATABASE_URL (for host extraction)")

if missing_vars:
    print("ERROR: The following environment variables are missing:")
    for var in missing_vars:
        print(f"- {var}")
    exit(1) # Exit if essential config is missing

# Construct DSN (Data Source Name) for psycopg
DB_CONN_INFO = f"dbname='{POSTGRES_DB_NAME}' user='{POSTGRES_USER}' password='{POSTGRES_PASSWORD}' host='{POSTGRES_HOST}' port='{POSTGRES_PORT}'"

# --- Helper Functions ---

def get_aws_client():
    """Creates a boto3 client configured for AWS S3."""
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            endpoint_url=AWS_ENDPOINT # Will use default AWS endpoint if None
        )
        client.head_bucket(Bucket=AWS_BUCKET_NAME)
        print("Successfully connected to AWS S3.")
        return client
    except ClientError as e:
        print(f"ERROR: Failed to connect to AWS S3 bucket '{AWS_BUCKET_NAME}'. Check credentials/permissions/endpoint.")
        print(f"Boto3 ClientError: {e}")
        exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred connecting to AWS S3: {e}")
        exit(1)

# Removed get_db_connection as connections are made per worker now

def list_s3_domains(s3_client, bucket: str | None, base_prefix: str) -> List[str]:
    """Lists the common prefixes (domains) directly under the base_prefix."""
    domain_prefixes = []
    paginator = s3_client.get_paginator('list_objects_v2')
    print(f"Listing domains (common prefixes) in s3://{bucket}/{base_prefix}...")
    try:
        # Ensure base_prefix ends with '/' for correct delimiter behavior
        if not base_prefix.endswith('/'):
            base_prefix += '/'

        page_iterator = paginator.paginate(Bucket=bucket, Prefix=base_prefix, Delimiter='/')
        for page in page_iterator:
            if "CommonPrefixes" in page:
                for prefix_info in page["CommonPrefixes"]:
                    domain_prefixes.append(prefix_info["Prefix"]) # e.g., "dla-dataset/financial/"
        print(f"Found {len(domain_prefixes)} potential domain prefixes.")
        return domain_prefixes
    except ClientError as e:
        print(f"ERROR: Failed to list domains in bucket '{bucket}' with prefix '{base_prefix}'.")
        print(f"Boto3 ClientError: {e}")
        return []
    except Exception as e:
        print(f"ERROR: An unexpected error occurred listing S3 domains: {e}")
        return []

def download_and_parse_json(s3_client, bucket: str | None, key: str) -> Optional[Dict[str, Any]]:
    """Downloads a JSON file from S3 and parses it."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        return data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"  S3 Error: Key not found - {key}")
        else:
            print(f"  S3 ClientError downloading {key}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  JSON Error: Failed to parse {key}: {e}")
        return None
    except Exception as e:
        print(f"  Unexpected Error downloading/parsing {key}: {e}")
        return None

def log_upload_failure(
    db_conn_info: str,
    s3_key: str,
    failure_reason: str,
    page_identifier: Optional[str] = None
) -> None:
    """
    Inserts a row into upload_failures. Uses its own autocommit connection
    so it won't interfere with the main ingest transactions.
    """
    try:
        # each call uses a fresh autocommit connection
        with psycopg.connect(db_conn_info) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO upload_failures (
                        page_identifier,
                        s3_key,
                        failure_reason
                    ) VALUES (%s, %s, %s);
                    """,
                    (page_identifier, s3_key, failure_reason)
                )
    except Exception as e:
        # best‐effort logging; do not raise
        print(f"[upload_failures] ERROR logging failure for key={s3_key}: {e}")

# --- Single Key Worker Function (Processes one JSON file) ---
def process_s3_key_worker(
    s3_key: str,
    s3_client, # Pass client instead of creating one per worker
    aws_bucket: str | None,
    db_conn_info: str # Pass connection string
) -> Tuple[str, bool, str]:
    """
    Worker task: Download JSON, parse, connect to DB, insert data for a SINGLE key.
    Returns (s3_key, success_status, message)
    """
    # Extract page_identifier assuming structure like '.../domain/page_id/extracted_data.json'
    try:
        page_id = Path(s3_key).parent.name
    except Exception:
        page_id = "unknown_page" # Fallback if path structure is unexpected
    log_prefix = f"[{page_id} | {s3_key[-30:]}]" # Short identifier for logs

    # 1. Download and Parse JSON
    json_data = download_and_parse_json(s3_client, aws_bucket, s3_key)
    if json_data is None:
        return s3_key, False, "Failed to download or parse JSON"

    # 2. Validate expected structure (basic check)
    required_keys = ["page_identifier", "domain", "source", "orientation", "color_mode",
                     "complexity", "layout_style", "primary_language", "segments",
                     "defects", "is_sparse_or_empty"]
    if not all(key in json_data for key in required_keys):
        missing = [key for key in required_keys if key not in json_data]
        return s3_key, False, f"JSON missing required keys: {missing}"

    # Extract data
    page_identifier = json_data.get("page_identifier")
    domain = json_data.get("domain")
    source = json_data.get("source")
    orientation = json_data.get("orientation")
    color_mode = json_data.get("color_mode")
    complexity = json_data.get("complexity")
    layout_style = json_data.get("layout_style")
    primary_language = json_data.get("primary_language")
    segments = json_data.get("segments", [])
    defects = json_data.get("defects", [])
    is_sparse_or_empty = json_data.get("is_sparse_or_empty")
    aws_s3_image_key = s3_key.replace("extracted_data.json", "source_image.jpg") # Derive image key

    # Ensure page_identifier from JSON matches the one derived from S3 key path
    if page_identifier != page_id:
         print(f"{log_prefix} WARNING: Mismatch between page_identifier in JSON ('{page_identifier}') and derived from S3 key ('{page_id}'). Using JSON value.")
         if page_identifier is None:
             return s3_key, False, "page_identifier is null in JSON"

    # 3. Connect to DB and Insert Data
    conn = None
    try:
        conn = psycopg.connect(db_conn_info)
        conn.autocommit = False

        with conn.cursor() as cur:
            # Insert into page_profiles
            sql_profile = """
                INSERT INTO page_profiles (
                    page_identifier, domain, source, orientation, color_mode,
                    complexity, layout_style, primary_language, is_sparse_or_empty,
                    aws_s3_json_key, aws_s3_image_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (page_identifier) DO NOTHING;
            """
            cur.execute(sql_profile, (
                page_identifier, domain, source, orientation, color_mode,
                complexity, layout_style, primary_language, is_sparse_or_empty,
                s3_key, aws_s3_image_key
            ))
            profile_inserted = cur.rowcount > 0

            # Insert into page_segments
            if segments:
                sql_segment = """
                    INSERT INTO page_segments (page_identifier, segment_type)
                    VALUES (%s, %s) ON CONFLICT (page_identifier, segment_type) DO NOTHING;
                """
                segment_data = [(page_identifier, seg) for seg in segments]
                cur.executemany(sql_segment, segment_data)

            # Insert into page_defects
            if defects:
                sql_defect = """
                    INSERT INTO page_defects (page_identifier, defect_type)
                    VALUES (%s, %s) ON CONFLICT (page_identifier, defect_type) DO NOTHING;
                """
                defect_data = [(page_identifier, defect) for defect in defects]
                cur.executemany(sql_defect, defect_data)

            conn.commit()
            status_msg = "Inserted" if profile_inserted else "Skipped (Already Exists)"
            return s3_key, True, status_msg

    except psycopg.Error as e:
        # print(f"{log_prefix} DB Error: {e}") # Can be very noisy
        if conn: conn.rollback()
        return s3_key, False, f"Database Error: {e}"
    except Exception as e:
        # print(f"{log_prefix} Unexpected Worker Error: {e}") # Can be noisy
        if conn: conn.rollback()
        return s3_key, False, f"Unexpected Worker Error: {e}"
    finally:
        if conn: conn.close()


# --- Domain Worker Function (Processes all keys within a domain using batches) ---
def process_domain_worker(
    domain_prefix: str,
    s3_client, # Pass client
    aws_bucket: str | None,
    db_conn_info: str,
    num_batch_workers: int, # Keep parameter, but will be passed fixed value (5)
    batch_size: int = 1000 # How many keys to list per S3 API call (max 1000)
) -> Tuple[str, int, int, int, int, int, int]:
    """
    Processes all extracted_data.json files under a specific domain prefix using batched sub-workers.
    Returns: (domain_prefix, processed, inserted, skipped, s3_failed, db_failed, other_failed)
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    processed_count = 0
    inserted_count = 0
    skipped_count = 0
    s3_parse_failed_count = 0
    db_failed_count = 0
    other_failed_count = 0
    batch_num = 0
    domain_name = domain_prefix.split('/')[-2] # Get domain name for logging

    print(f"[{domain_name}] Starting processing...") # Keep: Domain start
    domain_start_time = time.time()

    try:
        # Paginate through objects within the domain prefix
        for page in paginator.paginate(Bucket=aws_bucket, Prefix=domain_prefix, PaginationConfig={'PageSize': batch_size}):
            batch_num += 1
            keys_in_batch = []
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Ensure it's the target file and not the prefix itself
                    if key.endswith("extracted_data.json") and key != domain_prefix:
                        keys_in_batch.append(key)

            if not keys_in_batch:
                # Log skipping empty batch if desired, or just continue
                # print(f"[{domain_name}] Skipping empty page/batch {batch_num}.")
                continue # Skip empty pages

            batch_process_start_time = time.time() # For batch duration calculation

            # Use a nested ThreadPoolExecutor for the current batch
            thread_name = f"{domain_name[:15]}_B{batch_num}_W"
            with ThreadPoolExecutor(max_workers=num_batch_workers, thread_name_prefix=thread_name) as batch_executor:
                future_to_key_batch: Dict[Any, str] = {}
                for key in keys_in_batch:
                    future = batch_executor.submit(
                        process_s3_key_worker, # Use the single-key worker
                        key, s3_client, aws_bucket, db_conn_info
                    )
                    future_to_key_batch[future] = key

                # Process results as they complete within the batch
                batch_processed_in_this = 0
                batch_inserted = 0
                batch_skipped = 0
                batch_s3_failed = 0
                batch_db_failed = 0
                batch_other_failed = 0

                for future in as_completed(future_to_key_batch):
                    processed_count += 1 # Increment overall domain counter
                    batch_processed_in_this += 1 # Increment this batch's counter
                    original_key = future_to_key_batch[future]
                    try:
                        key_result, success, status_msg = future.result()
                        if success:
                            if status_msg == "Inserted":
                                inserted_count += 1
                                batch_inserted += 1
                            else: # Skipped
                                skipped_count += 1
                                batch_skipped += 1
                        else:
                            # derive page_id from path if JSON didn't supply one
                            page_id = Path(original_key).parent.name
                            # log into upload_failures table
                            log_upload_failure(
                                db_conn_info=db_conn_info,
                                s3_key=original_key,
                                failure_reason=status_msg,
                                page_identifier=page_id
                            )

                            # existing categorization logic
                            if "download or parse" in status_msg or "JSON missing" in status_msg:
                                s3_parse_failed_count += 1
                                batch_s3_failed += 1
                            elif "Database Error" in status_msg:
                                db_failed_count += 1
                                batch_db_failed += 1
                            else:
                                other_failed_count += 1
                                batch_other_failed += 1

                    except Exception as exc:
                        # also log unexpected future‐level errors
                        page_id = Path(original_key).parent.name
                        err_text = f"System Error: {exc}"
                        log_upload_failure(
                            db_conn_info=db_conn_info,
                            s3_key=original_key,
                            failure_reason=err_text,
                            page_identifier=page_id
                        )

                        other_failed_count += 1
                        batch_other_failed += 1
                        print(f"[{domain_name}] System Error processing future in batch {batch_num} for key {original_key}: {exc}")

            batch_duration = time.time() - batch_process_start_time
            # Log after EVERY batch completion
            print(f"[{domain_name}] Batch {batch_num} completed in {batch_duration:.2f}s. "
                  f"Batch Stats: Processed={batch_processed_in_this}, Inserted={batch_inserted}, Skipped={batch_skipped}, "
                  f"Failed(S3/Parse)={batch_s3_failed}, Failed(DB)={batch_db_failed}, Failed(Other)={batch_other_failed}. "
                  f"Domain Total Processed: {processed_count}")


    except ClientError as e:
         # Keep: Essential error for S3 issues
         print(f"[{domain_name}] S3 Error during pagination: {e}. Stopping processing for this domain.")
    except Exception as e:
        # Keep: Essential error for unexpected issues
        print(f"[{domain_name}] Unexpected error during processing: {e}. Stopping processing for this domain.")
        import traceback
        traceback.print_exc()

    domain_duration = time.time() - domain_start_time
    # Keep: Domain completion summary
    print(f"[{domain_name}] Finished processing domain in {domain_duration:.2f}s. "
          f"Total Processed: {processed_count}, Inserted: {inserted_count}, Skipped: {skipped_count}, "
          f"Failed (S3/Parse): {s3_parse_failed_count}, Failed (DB): {db_failed_count}, Failed (Other): {other_failed_count}")

    return (domain_prefix, processed_count, inserted_count, skipped_count, s3_parse_failed_count, db_failed_count, other_failed_count)


# --- Database Backup Function ---
def backup_database(db_name, user, host, port, backup_file_path, password=None):
    """
    Performs a backup of the PostgreSQL database using pg_dump.
    """
    # Create backup directory if it doesn't exist
    Path(backup_file_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'pg_dump',
        '--dbname', db_name,
        '--username', user,
        '--host', host,
        '--port', str(port),
        '--format=custom',  # Custom format is generally good for pg_restore
        '--blobs',          # Include large objects
        '--verbose',        # Optional: for more output during backup
        f'--file={backup_file_path}'
    ]

    # Set PGPASSWORD environment variable for the subprocess
    env = os.environ.copy()
    if password:
        env['PGPASSWORD'] = password

    try:
        print(f"Starting database backup to {backup_file_path}...")
        # Construct a command string for printing that hides the password if it were part of the command
        # (though we are using PGPASSWORD env var, this is good practice if cmd structure changes)
        # For pg_dump, password is not directly in cmd args when using PGPASSWORD.
        print(f"Executing command: pg_dump --dbname={db_name} --username={user} --host={host} --port={str(port)} -F c -b -v --file={backup_file_path}")

        process = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print("Database backup successful.")
        # pg_dump often uses stderr for progress messages even on success
        if process.stderr:
            print("pg_dump messages (stderr):\n", process.stderr)
        if process.stdout: # stdout might also contain info with -v
            print("pg_dump output (stdout):\n", process.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Database backup failed. pg_dump exited with code: {e.returncode}")
        if e.stdout:
            print("Stdout:\n", e.stdout)
        if e.stderr:
            print("Stderr:\n", e.stderr)
        return False
    except FileNotFoundError:
        print("ERROR: pg_dump command not found. Please ensure PostgreSQL client tools are installed and in your PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during backup: {e}")
        return False

# --- Main Execution ---
def run_upload(): # Removed worker args from function signature
    # --- Fixed Worker Counts ---
    TARGET_DOMAIN_WORKERS = 21
    NUM_BATCH_WORKERS = 5 # Fixed at 5

    print("-" * 40)
    print(f"Starting PostgreSQL Upload Pipeline (Domain/Batch Strategy)")
    # Updated print statement for fixed worker counts
    print(f"Targeting {TARGET_DOMAIN_WORKERS} concurrent domain workers (actual may be lower if fewer domains found).")
    print(f"Using {NUM_BATCH_WORKERS} sub-workers per domain batch.")
    print(f"Target Bucket: s3://{AWS_BUCKET_NAME}/{AWS_S3_BASE_PREFIX}")
    print(f"Target DB: {POSTGRES_DB_NAME} on {POSTGRES_HOST}:{POSTGRES_PORT}")
    print("-" * 40)

    start_time = time.time()
    # Overall counters across all domains
    overall_processed_count = 0
    overall_inserted_count = 0
    overall_skipped_count = 0
    overall_failed_s3_parse = 0
    overall_failed_db = 0
    overall_failed_other = 0
    domains_completed = 0
    domains_failed_entirely = 0

    s3_client = get_aws_client() # Initialize S3 client once

    # --- Get Domains to Process ---
    domain_prefixes = list_s3_domains(s3_client, AWS_BUCKET_NAME, AWS_S3_BASE_PREFIX)
    total_domains = len(domain_prefixes)
    if total_domains == 0:
        print("No domain prefixes found to process. Exiting.")
        return

    # Adjust domain workers if fewer domains are found than the target
    actual_domain_workers = min(total_domains, TARGET_DOMAIN_WORKERS)
    if actual_domain_workers < TARGET_DOMAIN_WORKERS:
        print(f"INFO: Found {total_domains} domains, using {actual_domain_workers} domain workers instead of {TARGET_DOMAIN_WORKERS}.")
    else:
        print(f"Using {actual_domain_workers} domain workers.")


    print(f"Submitting tasks for {total_domains} domains to the main executor...")

    # --- ThreadPoolExecutor for Domains ---
    future_to_domain: Dict[Any, str] = {}

    try:
        # Outer executor manages domain workers, using the adjusted count
        with ThreadPoolExecutor(max_workers=actual_domain_workers, thread_name_prefix='DomainWorker') as domain_executor:
            for prefix in domain_prefixes:
                future = domain_executor.submit(
                    process_domain_worker,
                    prefix,
                    s3_client,
                    AWS_BUCKET_NAME,
                    DB_CONN_INFO,
                    NUM_BATCH_WORKERS # Pass fixed number of batch workers
                )
                future_to_domain[future] = prefix

            print("All domain tasks submitted. Waiting for completion...")

            # Process results as domains complete
            for future in as_completed(future_to_domain):
                domain = future_to_domain[future]
                domain_name_log = domain.split('/')[-2]
                try:
                    # Result tuple: (prefix, processed, inserted, skipped, s3_failed, db_failed, other_failed)
                    d_prefix, d_processed, d_inserted, d_skipped, d_s3, d_db, d_other = future.result()
                    domains_completed += 1

                    # Update overall counters
                    overall_processed_count += d_processed
                    overall_inserted_count += d_inserted
                    overall_skipped_count += d_skipped
                    overall_failed_s3_parse += d_s3
                    overall_failed_db += d_db
                    overall_failed_other += d_other

                    # Keep: Print summary when a domain finishes (already present)
                    print(f"\n--- Domain '{domain_name_log}' COMPLETED ---")
                    print(f"  Processed: {d_processed} | Inserted: {d_inserted} | Skipped: {d_skipped}")
                    print(f"  Failed S3/Parse: {d_s3} | Failed DB: {d_db} | Failed Other: {d_other}")
                    print(f"--- Overall Progress: {domains_completed}/{total_domains} domains completed ---")

                except Exception as exc:
                    domains_failed_entirely += 1
                    # Keep: Essential error for domain worker failure (already present)
                    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"Domain '{domain_name_log}' worker FAILED entirely: {exc}")
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                    # Optionally log traceback here
                    # import traceback
                    # traceback.print_exc()

    except Exception as e:
        # Keep: Essential error for main loop failure (already present)
        print(f"An unexpected error occurred in the main execution loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Final Summary ---
        end_time = time.time()
        total_elapsed = end_time - start_time
        print("-" * 50)
        print("Upload Pipeline Finished.")
        print(f"  Domains Processed: {domains_completed}/{total_domains}")
        if domains_failed_entirely > 0:
             print(f"  Domains Failed Entirely: {domains_failed_entirely}")
        print("-" * 20)
        print(f"  Overall Keys Processed Attempts: {overall_processed_count}")
        print(f"  Overall Successfully Inserted: {overall_inserted_count}")
        print(f"  Overall Skipped (Already Existed): {overall_skipped_count}")
        print(f"  Overall Failed (S3 Download/Parse): {overall_failed_s3_parse}")
        print(f"  Overall Failed (Database): {overall_failed_db}")
        print(f"  Overall Failed (Other/Worker): {overall_failed_other}")
        print("-" * 20)
        print(f"  Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
        # Calculate rate based on processed keys, might not be super accurate if domains vary wildly
        final_rate = overall_processed_count / total_elapsed if total_elapsed > 0 else 0
        print(f"  Approx. Average rate: {final_rate:.2f} keys/sec")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload DLA data to PostgreSQL or backup the database.")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Perform a database backup instead of uploading data. Uses POSTGRES_* and AWS_* env vars for configuration."
    )
    args = parser.parse_args()

    if args.backup:
        print("Backup mode selected.")
        Path(POSTGRES_BACKUP_DIR).mkdir(parents=True, exist_ok=True) # Ensure backup directory exists

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Using .dump extension, common for pg_dump custom format
        backup_file_name = f"{POSTGRES_DB_NAME}_backup_{timestamp}.dump"
        backup_file_path = Path(POSTGRES_BACKUP_DIR) / backup_file_name

        # Environment variables for DB connection are validated at the start of the script.
        # If any are missing, the script would have exited.
        # We re-check here for clarity before calling backup.
        required_db_vars = {
            "POSTGRES_DB_NAME": POSTGRES_DB_NAME,
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_HOST": POSTGRES_HOST,
            "POSTGRES_PORT": POSTGRES_PORT
        }
        missing_backup_vars = [k for k, v in required_db_vars.items() if not v]
        if missing_backup_vars:
            print("ERROR: Missing one or more PostgreSQL connection environment variables for backup:")
            for var_name in missing_backup_vars:
                print(f"- {var_name}")
            exit(1)

        success = backup_database(
            db_name=POSTGRES_DB_NAME,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD, # Passed to set PGPASSWORD
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            backup_file_path=str(backup_file_path)
        )
        if success:
            print(f"Backup completed: {backup_file_path}")
        else:
            print("Backup failed.")
            exit(1)
    else:
        # Directly call run_upload as worker counts are handled internally
        run_upload()