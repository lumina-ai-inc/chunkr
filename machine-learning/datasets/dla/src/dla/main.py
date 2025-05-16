import os
from pathlib import Path
from dotenv import load_dotenv
import time
# Import ThreadPoolExecutor and as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Settings
from .downloader import get_s3_client, list_page_keys, parse_key, download_and_hash
from .database import Database

# --- Worker Function ---
def process_key_worker(client, bucket_name, key):
    """Parses, downloads, hashes a single key. Returns results or None."""
    meta = parse_key(key)
    if not meta:
        return key, None # Return key and None result if parse fails

    page_id, h = download_and_hash(client, bucket_name, key)
    if page_id is None or h is None:
        return key, None # Return key and None result if download/hash fails

    # Return key and tuple of results needed for DB insert
    return key, (page_id, meta[0], meta[1], meta[2], h, key)
# --- End Worker Function ---

def run():
    # Calculate the expected path to the .env file
    dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'

    # Debugging: Check if the .env file exists
    print(f"Attempting to load .env file from: {dotenv_path}")
    if dotenv_path.is_file():
        print(".env file found. Loading variables...")
        loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
        print(f".env file loaded: {loaded}") # Check if load_dotenv returns True
    else:
        print(f"WARNING: .env file not found at {dotenv_path}. Relying on environment variables.")
        # raise FileNotFoundError(f".env file not found at {dotenv_path}")

    # Explicitly get variables after attempting to load .env
    access_key = os.getenv("GOOGLE_ACCESS_KEY")
    secret_key = os.getenv("GOOGLE_SECRET_KEY")
    endpoint = os.getenv("GOOGLE_ENDPOINT")
    bucket = os.getenv("GOOGLE_BUCKET_NAME")

    # Check if variables were loaded successfully
    missing_vars = []
    if not access_key: missing_vars.append("GOOGLE_ACCESS_KEY")
    if not secret_key: missing_vars.append("GOOGLE_SECRET_KEY")
    if not endpoint: missing_vars.append("GOOGLE_ENDPOINT")
    if not bucket: missing_vars.append("GOOGLE_BUCKET_NAME")

    if missing_vars:
        print(f"ERROR: The following environment variables are missing or empty after attempting to load .env:")
        for var in missing_vars:
            print(f"- {var}")
        print(f"Please ensure they are set in the environment or in the .env file at: {dotenv_path}")
        return # Exit if variables are missing

    # Add assertions to assure the type checker that variables are not None
    assert access_key is not None, "GOOGLE_ACCESS_KEY is None after check"
    assert secret_key is not None, "GOOGLE_SECRET_KEY is None after check"
    assert endpoint is not None, "GOOGLE_ENDPOINT is None after check"
    assert bucket is not None, "GOOGLE_BUCKET_NAME is None after check"

    try:
        settings = Settings(
            google_access_key=access_key,
            google_secret_key=secret_key,
            google_endpoint=endpoint,
            bucket_name=bucket
        )
    except Exception as e:
        print(f"Error initializing Settings even after variables seemed loaded: {e}")
        return

    # Use a try/finally block to ensure DB connection is closed
    db = None # Initialize db to None
    try:
        client = get_s3_client(settings)
        db = Database("raw_and_dedup.db")
        db.create_tables()

        print("Starting metadata fetching, hashing, and DB population (concurrent)...")
        count = 0
        error_count = 0
        parse_error_count = 0
        total_keys_submitted = 0
        start_time = time.time()
        last_print_time = start_time
        last_commit_time = start_time
        print_interval_seconds = 30 # Print progress more often
        commit_interval_seconds = 10 # Commit DB changes every 10 seconds

        # --- ThreadPoolExecutor Setup ---
        # Adjust max_workers based on testing. Start with a moderate number.
        # Too many can overwhelm network or CPU, too few limits concurrency.
        # Good starting point: 2x-5x number of CPU cores, or based on network limits.
        # Let's start with 50 for significant I/O overlap.
        num_workers = 50
        # Limit the number of tasks submitted ahead of time to avoid excessive memory use
        max_queued_tasks = num_workers * 4

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {} # Dictionary to map Future object to its key

            key_iterator = list_page_keys(client, settings.bucket_name)
            keys_finished = False

            while not keys_finished or futures:
                # --- Submit new tasks ---
                # Fill the queue up to max_queued_tasks
                while not keys_finished and len(futures) < max_queued_tasks:
                    try:
                        key = next(key_iterator)
                        future = executor.submit(process_key_worker, client, settings.bucket_name, key)
                        futures[future] = key # Store future -> key mapping
                        total_keys_submitted += 1
                    except StopIteration:
                        keys_finished = True # No more keys left
                        print("All keys submitted to workers.")
                        break # Exit submission loop
                    except Exception as e:
                        print(f"Error submitting key {key}: {e}") # Handle potential submission errors
                        error_count += 1

                # --- Process completed tasks ---
                # Wait for at least one future to complete, with a timeout
                # to allow periodic checks/commits even if workers are slow.
                done_futures = set()
                try:
                    # Use as_completed with a timeout
                    for future in as_completed(futures, timeout=1.0):
                        key = futures.pop(future) # Remove from tracking dict
                        done_futures.add(future) # Mark as processed in this batch
                        try:
                            _, result_data = future.result() # Get worker result (key, data_tuple)
                            if result_data:
                                db.insert_raw(*result_data) # Unpack tuple into insert_raw args
                                count += 1
                            else:
                                # Handle cases where worker returned None (parse or download error)
                                # We could differentiate parse vs download errors if needed
                                parse_error_count += 1 # Assume parse error for simplicity now
                        except Exception as e:
                            print(f"  Error processing result for key {key}: {e}")
                            error_count += 1
                except TimeoutError:
                    # No futures completed within the timeout, just continue the loop
                    pass

                # Remove processed futures from the main tracking dict (redundant with pop but safe)
                for future in done_futures:
                    if future in futures:
                        del futures[future]

                # --- Periodic Commit and Progress Update ---
                current_time = time.time()
                if current_time - last_commit_time >= commit_interval_seconds:
                    db.commit()
                    # print(f"DB commit at {count} pages.") # Optional debug print
                    last_commit_time = current_time

                if current_time - last_print_time >= print_interval_seconds:
                    elapsed_time = current_time - start_time
                    pages_per_second = count / elapsed_time if elapsed_time > 0 else 0
                    print(f"Progress: {count} success | {total_keys_submitted} submitted | "
                          f"{len(futures)} active | {parse_error_count} parse err | {error_count} other err | "
                          f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} | "
                          f"Rate: {pages_per_second:.2f} pages/sec")
                    last_print_time = current_time

            # --- End of main loop ---
            print("Main loop finished. Waiting for final tasks if any...")

        # Final commit after loop finishes
        print("Performing final DB commit...")
        db.commit()

        # Final summary print (similar to before)
        end_time = time.time()
        total_elapsed = end_time - start_time
        final_rate = count / total_elapsed if total_elapsed > 0 else 0
        print("-" * 40)
        print(f"Finished processing.")
        print(f"  Total pages added to raw DB: {count}")
        print(f"  Total keys submitted: {total_keys_submitted}")
        print(f"  Parse/Download Errors: {parse_error_count}")
        print(f"  Other Errors encountered: {error_count}")
        print(f"  Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
        print(f"  Average rate: {final_rate:.2f} pages/sec")
        print("-" * 40)

        print("Populating deduplication table...")
        db.populate_dedup()
        print("✅ Metadata fetched, hashed, and deduplicated into raw_and_dedup.db")

    except Exception as e:
        print(f"An unexpected error occurred in the main execution: {e}")
        # Optionally re-raise the exception if needed for debugging
        # raise e
    finally:
        # Ensure database connection is closed even if errors occur
        if db:
            print("Closing database connection.")
            db.close()

if __name__ == "__main__":
    run()