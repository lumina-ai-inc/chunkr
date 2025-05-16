import os
import sqlite3
import psycopg # For PostgreSQL
from psycopg import sql
import random
import math
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any

# --- Configuration ---
# Assumes the SQLite DB is in the current directory or where the script is run from.
# This matches the DB_PATH from your extract_and_upload.py if run from the project root.
SQLITE_DB_PATH = Path("raw_and_dedup.db").resolve()

# User IDs and the fraction of their documents to KEEP
USER_SAMPLES_CONFIG = {
    "5178361f-c049-4ab3-86f6-48a5d1620588": 7/10,
}

# Load .env file from the current directory (or parent)
# Ensure your .env file contains DATABASE_URL for PostgreSQL
print(f"Attempting to load .env file from current directory or parent directories...")
if load_dotenv(override=True):
    print(".env file loaded successfully.")
else:
    print("WARNING: .env file not found or not loaded. Relying on pre-set environment variables for DATABASE_URL.")

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_sqlite_connection(db_path: Path) -> sqlite3.Connection:
    """Establishes a connection to the SQLite database."""
    if not db_path.is_file():
        raise FileNotFoundError(f"SQLite database file not found at {db_path}")
    conn = sqlite3.connect(db_path)
    print(f"Successfully connected to SQLite database: {db_path.name}")
    return conn

def get_pg_connection(conn_string: str | None) -> psycopg.Connection | None:
    """Establishes a connection to the PostgreSQL database."""
    if not conn_string:
        print("ERROR: DATABASE_URL environment variable is not set. Cannot connect to PostgreSQL.")
        return None
    try:
        conn = psycopg.connect(conn_string)
        print("Successfully connected to PostgreSQL database.")
        return conn
    except psycopg.Error as e:
        print(f"ERROR: Unable to connect to PostgreSQL using DATABASE_URL: {e}")
        return None

def get_doc_to_page_ids_for_user_sqlite(
    sqlite_cursor: sqlite3.Cursor, user_id: str
) -> Dict[str, List[str]]:
    """
    Fetches all doc_ids for a given user_id from SQLite pages_raw,
    and maps each doc_id to a list of its associated page_identifiers (dedup_ids).
    """
    sqlite_cursor.execute("""
        SELECT doc_id, dedup_id FROM pages_raw
        WHERE user_id = ? AND dedup_id IS NOT NULL
    """, (user_id,))
    
    doc_to_page_ids: Dict[str, List[str]] = {}
    for doc_id, page_id in sqlite_cursor.fetchall():
        if doc_id not in doc_to_page_ids:
            doc_to_page_ids[doc_id] = []
        doc_to_page_ids[doc_id].append(page_id)
    return doc_to_page_ids

def sample_doc_ids(doc_ids: list[str], fraction_to_keep: float) -> list[str]:
    """Randomly samples a fraction of document IDs."""
    if not doc_ids:
        return []
    num_to_keep = int(round(len(doc_ids) * fraction_to_keep))
    num_to_keep = max(0, min(len(doc_ids), num_to_keep)) # Ensure k is valid
    return random.sample(doc_ids, num_to_keep)

def filter_user_documents_pg(
    sqlite_db_path: Path,
    user_configs: Dict[str, float],
    pg_conn_string: str | None,
    dry_run: bool = True
):
    """
    Filters documents for specified users in the PostgreSQL database.
    It uses SQLite to map user_id/doc_id to page_identifiers (dedup_ids),
    then deletes corresponding entries from PostgreSQL's page_profiles table.
    """
    if not pg_conn_string:
        print("PostgreSQL connection string (DATABASE_URL) is missing. Aborting.")
        return

    sqlite_conn = None
    pg_conn = None

    try:
        sqlite_conn = get_sqlite_connection(sqlite_db_path)
        pg_conn = get_pg_connection(pg_conn_string)

        if not sqlite_conn or not pg_conn:
            print("Failed to establish one or more database connections. Aborting.")
            return

        sqlite_cursor = sqlite_conn.cursor()
        
        print(f"\n--- Starting Document Filtering in PostgreSQL {'(DRY RUN)' if dry_run else '(LIVE RUN)'} ---")
        print(f"Using SQLite DB: {sqlite_db_path.name} for user/doc/page_id mapping.")
        print(f"Targeting PostgreSQL DB (via DATABASE_URL).")

        total_pg_rows_affected_overall = 0

        with pg_conn.cursor() as pg_cursor:
            for user_id, fraction in user_configs.items():
                print(f"\nProcessing user_id: {user_id}")

                doc_to_page_ids_map = get_doc_to_page_ids_for_user_sqlite(sqlite_cursor, user_id)
                
                if not doc_to_page_ids_map:
                    print(f"  No documents (or no dedup_ids) found for user {user_id} in SQLite DB. Skipping.")
                    continue

                original_doc_ids = list(doc_to_page_ids_map.keys())
                num_original_docs = len(original_doc_ids)
                print(f"  Found {num_original_docs} unique documents (doc_ids) for user {user_id} in SQLite DB.")

                doc_ids_to_keep = sample_doc_ids(original_doc_ids, fraction)
                num_docs_to_keep = len(doc_ids_to_keep)
                print(f"  Targeting to keep {num_docs_to_keep} documents (approx. {fraction*100:.1f}% of original).")

                doc_ids_to_remove = [doc_id for doc_id in original_doc_ids if doc_id not in doc_ids_to_keep]
                num_docs_to_remove = len(doc_ids_to_remove)

                if not doc_ids_to_remove:
                    print(f"  No documents selected for removal for user {user_id}. Skipping deletion phase.")
                    continue
                
                print(f"  Identified {num_docs_to_remove} documents (doc_ids) for removal for user {user_id}.")

                page_ids_to_delete: List[str] = []
                for doc_id in doc_ids_to_remove:
                    page_ids_to_delete.extend(doc_to_page_ids_map.get(doc_id, []))
                
                num_page_ids_to_delete = len(page_ids_to_delete)

                if not page_ids_to_delete:
                    print(f"  No page_identifiers (dedup_ids) found for the documents marked for removal. Skipping deletion.")
                    continue
                
                print(f"  These {num_docs_to_remove} documents correspond to {num_page_ids_to_delete} page_identifiers (dedup_ids).")
                
                # Count how many rows in page_profiles would be affected
                # Using = ANY(%s) which expects a list or tuple as the parameter
                count_query = sql.SQL("SELECT COUNT(*) FROM page_profiles WHERE page_identifier = ANY(%s)")
                pg_cursor.execute(count_query, (page_ids_to_delete,))
                row = pg_cursor.fetchone() # Fetch the row first
                num_pg_rows_to_delete = row[0] if row else 0 # Check if row is not None before subscripting

                if num_pg_rows_to_delete == 0:
                    print(f"  No matching entries found in PostgreSQL 'page_profiles' for the {num_page_ids_to_delete} page_identifiers. Skipping deletion.")
                    continue
                
                print(f"  Found {num_pg_rows_to_delete} entries in 'page_profiles' to be deleted for user {user_id}.")

                if dry_run:
                    print(f"  [DRY RUN] Would delete {num_pg_rows_to_delete} entries from 'page_profiles' in PostgreSQL.")
                    total_pg_rows_affected_overall += num_pg_rows_to_delete
                else:
                    print(f"  Executing DELETE for {num_pg_rows_to_delete} entries from 'page_profiles' in PostgreSQL...")
                    delete_query = sql.SQL("DELETE FROM page_profiles WHERE page_identifier = ANY(%s)")
                    pg_cursor.execute(delete_query, (page_ids_to_delete,))
                    deleted_pg_rows_for_user = pg_cursor.rowcount
                    print(f"  Successfully deleted {deleted_pg_rows_for_user} entries from 'page_profiles' for user {user_id}.")
                    total_pg_rows_affected_overall += deleted_pg_rows_for_user
            
            if not dry_run and total_pg_rows_affected_overall > 0:
                print("\nCommitting changes to PostgreSQL database...")
                pg_conn.commit()
                print("Changes committed to PostgreSQL.")
            elif not dry_run and total_pg_rows_affected_overall == 0:
                print("\nNo changes were made to PostgreSQL that required committing.")
            else: # dry_run is True
                print("\n[DRY RUN] No changes were made to the PostgreSQL database.")

        print(f"\n--- PostgreSQL Filtering Summary ---")
        action_verb = "would be" if dry_run else "were"
        print(f"Total 'page_profiles' entries that {action_verb} deleted across all processed users: {total_pg_rows_affected_overall}")
        print(f"Associated entries in 'page_segments' and 'page_defects' {action_verb} also removed due to ON DELETE CASCADE.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except psycopg.Error as e:
        print(f"PostgreSQL Error: {e}")
        if pg_conn and not dry_run:
            print("Rolling back PostgreSQL transaction due to error.")
            pg_conn.rollback()
    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        if pg_conn and not dry_run: # Attempt rollback on general error too
            try:
                pg_conn.rollback()
                print("Rolled back PostgreSQL transaction due to unexpected error.")
            except Exception as rb_e:
                print(f"Error during rollback: {rb_e}")
    finally:
        if sqlite_conn:
            sqlite_conn.close()
            print("SQLite database connection closed.")
        if pg_conn:
            pg_conn.close()
            print("PostgreSQL database connection closed.")

if __name__ == "__main__":
    if not SQLITE_DB_PATH.is_file():
        print(f"SQLite Database file '{SQLITE_DB_PATH}' not found. Please ensure the path is correct and the file exists.")
    elif not DATABASE_URL:
        print("DATABASE_URL environment variable is not set. Please configure it in your .env file or environment.")
    else:
        print("This script will filter documents for specified users in your PostgreSQL database.")
        print("It uses the SQLite 'raw_and_dedup.db' to identify documents and their pages,")
        print("then reduces the number of documents by randomly sampling a subset to keep,")
        print("and finally deletes pages (from 'page_profiles') associated with the non-sampled documents for those users from PostgreSQL.")
        print("\nIMPORTANT: It is HIGHLY recommended to BACKUP BOTH your PostgreSQL database AND your 'raw_and_dedup.db' SQLite file before running in live mode.\n")

        run_mode_input = input("Run in DRY RUN mode (shows what would change, no actual deletion)? (yes/no) [default: yes]: ").strip().lower()
        is_dry_run = run_mode_input not in ['no', 'n']

        if is_dry_run:
            print("\nStarting in DRY RUN mode...")
            filter_user_documents_pg(SQLITE_DB_PATH, USER_SAMPLES_CONFIG, DATABASE_URL, dry_run=True)
        else:
            print("\nWARNING: You are about to run the script in LIVE mode.")
            print(f"This will permanently delete data from your PostgreSQL database and relies on '{SQLITE_DB_PATH.name}'.")
            confirm_live_run = input("Are you absolutely sure you want to proceed with live deletion? (yes/no): ").strip().lower()
            
            if confirm_live_run in ['yes', 'y']:
                print("\nStarting in LIVE mode...")
                filter_user_documents_pg(SQLITE_DB_PATH, USER_SAMPLES_CONFIG, DATABASE_URL, dry_run=False)
            else:
                print("Operation cancelled by user. No changes made.")