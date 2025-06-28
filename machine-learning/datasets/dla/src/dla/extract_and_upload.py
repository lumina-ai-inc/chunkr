import os
import base64
import json
import sqlite3
import time
import boto3
import requests # For calling OpenRouter
import argparse # <<< Add argparse import
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any, Optional
from pydantic import ValidationError

# Assuming schema.py is in the same directory or accessible
from .schema import (
    PageProfile,
    DocumentDomain,
    DocumentSource,
    Orientation,
    ColorMode,
    Complexity,
    LayoutStyle,
    SegmentType,
    DefectType,
    LanguageCode,
)
# Assuming config.py defines GCS settings
from .config import Settings as GcsSettings

# --- Configuration Loading ---
# Load environment variables from .env file located three levels up
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
print(f"Attempting to load .env file for extraction pipeline from: {dotenv_path}")
if dotenv_path.is_file():
    print(".env file found. Loading variables...")
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    print(f"WARNING: .env file not found at {dotenv_path}. Relying on environment variables.")

# --- AWS Configuration ---
AWS_ACCESS_KEY = os.environ.get("AWS__ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS__SECRET_KEY")
AWS_ENDPOINT = os.environ.get("AWS__ENDPOINT") # Optional, defaults to standard AWS endpoint if None
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", "chunkr-datasets") # Or your target bucket

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001") # Use the specific model tag if needed
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# --- Database Paths ---
DB_PATH = Path("raw_and_dedup.db").resolve() # Assumes it's in the root directory
FAILED_DB_PATH = Path("failed_extractions.db").resolve() # New DB for failures

# --- Concurrency & Timeout Settings ---
NUM_WORKERS = 50 # Start lower for LLM calls, adjust based on rate limits/performance
LLM_TIMEOUT_SECONDS = 120 # Timeout for the API call to OpenRouter

# --- Helper Functions ---

def get_gcs_settings() -> GcsSettings:
    """Loads GCS settings using pydantic-settings."""
    # Need to explicitly load GCS vars as pydantic-settings expects them
    # This assumes the .env was loaded correctly earlier
    return GcsSettings(
        google_access_key=os.environ.get("GOOGLE_ACCESS_KEY", ""),
        google_secret_key=os.environ.get("GOOGLE_SECRET_KEY", ""),
        google_endpoint=os.environ.get("GOOGLE_ENDPOINT", ""),
        bucket_name=os.environ.get("GOOGLE_BUCKET_NAME", "")
    )

def get_gcs_client(settings: GcsSettings):
    """Creates a boto3 client configured for GCS."""
    if not all([settings.google_access_key, settings.google_secret_key, settings.google_endpoint]):
         raise ValueError("Missing GCS configuration (key, secret, or endpoint).")
    return boto3.client(
        "s3",
        aws_access_key_id=settings.google_access_key,
        aws_secret_access_key=settings.google_secret_key,
        endpoint_url=settings.google_endpoint,
    )

def get_aws_client():
    """Creates a boto3 client configured for AWS."""
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        raise ValueError("Missing AWS__ACCESS_KEY or AWS__SECRET_KEY")
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        endpoint_url=AWS_ENDPOINT # Uses default AWS endpoint if None
    )

def get_unique_pages_from_db(db_path: Path) -> List[Tuple[str, str]]:
    """Queries the main DB for unique pages, returning (dedup_id, s3_key)."""
    unique_pages = []
    if not db_path.is_file():
        raise FileNotFoundError(f"Main database file not found at {db_path}. Run the initial download/dedup first.")
    try:
        # Use 'file:' URI for read-only connection if possible, prevents locking issues
        # db_uri = f'file:{db_path}?mode=ro'
        # with sqlite3.connect(db_uri, uri=True) as conn:
        # Simpler connection for now:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Get one s3_key for each distinct dedup_id that is not NULL
            cursor.execute("""
                SELECT dedup_id, MIN(s3_key)
                FROM pages_raw
                WHERE dedup_id IS NOT NULL
                GROUP BY dedup_id;
            """)
            unique_pages = cursor.fetchall()
            print(f"Found {len(unique_pages)} unique pages in the main database ({db_path.name}).")
    except sqlite3.Error as e:
        print(f"Database error while fetching unique pages from {db_path.name}: {e}")
        raise # Re-raise the error to stop execution
    return unique_pages

def get_failed_pages_from_db(db_path: Path) -> List[Tuple[str, str]]:
    """Queries the failed DB for pages to retry, returning (dedup_id, gcs_key)."""
    failed_pages = []
    if not db_path.is_file():
        print(f"Warning: Failed extractions database file not found at {db_path}. Cannot rerun failed pages.")
        return [] # Return empty list if DB doesn't exist
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT dedup_id, gcs_key
                FROM failed_pages;
            """)
            failed_pages = cursor.fetchall()
            if not failed_pages:
                print(f"No failed pages found in the database ({db_path.name}) to rerun.")
            else:
                print(f"Found {len(failed_pages)} failed pages in {db_path.name} to rerun.")
    except sqlite3.Error as e:
        print(f"Database error while fetching failed pages from {db_path.name}: {e}")
        # Decide if you want to raise or just return empty list
        return [] # Return empty on error to prevent script crash in rerun mode
    return failed_pages

def download_page_from_gcs(client, bucket_name: str, key: str) -> Optional[bytes]:
    """Downloads page content from GCS."""
    try:
        resp = client.get_object(Bucket=bucket_name, Key=key)
        return resp["Body"].read()
    except ClientError as e:
        # Check for specific errors like Not Found
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"  Error downloading GCS key {key}: Object not found.")
        else:
            print(f"  ClientError downloading GCS key {key}: {e}")
        return None
    except Exception as e:
        print(f"  Unexpected error downloading GCS key {key}: {e}")
        return None

def generate_llm_prompt() -> str:
    """Creates a detailed prompt for the LLM, using direct Enum definitions and providing context for all fields."""
    # --- Get enum values directly from imported classes ---
    source_values = [e.value for e in DocumentSource]
    orientation_values = [e.value for e in Orientation]
    color_mode_values = [e.value for e in ColorMode]
    complexity_values = [e.value for e in Complexity]
    layout_values = [e.value for e in LayoutStyle]
    segment_type_values = [e.value for e in SegmentType]
    defect_values = [e.value for e in DefectType]
    language_values = [e.value for e in LanguageCode]

    # --- Construct detailed descriptions from schema comments ---

    domain_descriptions = f"""
*   `{DocumentDomain.FINANCIAL.value}`: e.g., Annual reports, SEC filings, bank statements, loan applications, investment prospectuses.
*   `{DocumentDomain.BILLING.value}`: e.g., Invoices, purchase orders, receipts, utility bills, account statements.
*   `{DocumentDomain.TAX.value}`: e.g., Tax forms (W2, 1040, etc.), tax returns, official tax documents.
*   `{DocumentDomain.SUPPLY_CHAIN.value}`: e.g., Bills of lading, packing slips, shipping manifests, inventory reports, logistics documents, POs, PODs etc.
*   `{DocumentDomain.TECHNICAL.value}`: e.g., Manuals, specifications, datasheets, engineering drawings, code documentation.
*   `{DocumentDomain.RESEARCH.value}`: e.g., Academic papers, scientific articles, conference proceedings, thesis, study reports (excluding patents).
*   `{DocumentDomain.LEGAL.value}`: e.g., Contracts, court filings, legal briefs, NDAs, terms of service, deeds, wills.
*   `{DocumentDomain.GOVERNMENT.value}`: e.g., Official forms, regulations, public notices, legislative documents, census forms, government reports.
*   `{DocumentDomain.PROCUREMENT.value}`: e.g., Requests for Proposal (RFP), Requests for Information (RFI), Requests for Quotation (RFQ), Bids, Proposals, Grant Solicitations, Statements of Work (SOW).
*   `{DocumentDomain.CONSULTING.value}`: e.g., Presentations (slides), proposals, reports, case studies prepared by consulting firms.
*   `{DocumentDomain.MAGAZINE.value}`: e.g., Articles, layouts typical of popular or trade magazines (often multi-column, image-heavy).
*   `{DocumentDomain.NEWSPAPER.value}`: e.g., News articles, editorials, classifieds, typical newspaper layouts.
*   `{DocumentDomain.TEXTBOOK.value}`: e.g., Chapters from educational books, often with diagrams, exercises, specific formatting.
*   `{DocumentDomain.HISTORICAL.value}`: e.g., Archived documents, letters, manuscripts, old records, documents clearly pre-modern era.
*   `{DocumentDomain.PATENT.value}`: e.g., Official patent filings/grants with specific structure (abstract, claims, drawings).
*   `{DocumentDomain.EDUCATION.value}`: e.g., Homework assignments, exam papers, syllabi, lecture notes, educational worksheets (distinct from textbooks).
*   `{DocumentDomain.MEDICAL.value}`: e.g., Patient records, medical charts, prescriptions, hospital forms, medical billing (use 'billing' if primarily an invoice), EOBs (Explanation of Benefits). Excludes general research papers (use 'research').
*   `{DocumentDomain.REAL_ESTATE.value}`: e.g., Property listings, deeds (use 'legal' if primarily a legal contract), appraisal reports, lease agreements, MLS sheets.
*   `{DocumentDomain.CONSTRUCTION.value}`: e.g., Blueprints, architectural drawings, construction plans, permits, inspection reports, change orders, material lists, safety manuals specific to construction sites.
*   `{DocumentDomain.MISCELLANEOUS.value}`: e.g., Identification cards, resumes, certificates, flyers, brochures, menus, general correspondence not fitting other categories.
*   `{DocumentDomain.UNKNOWN.value}`: Fallback if no other category strongly fits.
"""

    source_descriptions = f"""
*   `{DocumentSource.NATIVE_DIGITAL.value}`: Born-digital (e.g., exported PDF, Word doc).
*   `{DocumentSource.SCANNED_CLEAN.value}`: Scanned from high-quality print.
*   `{DocumentSource.SCANNED_DEGRADED.value}`: Scanned from fax, old paper, noisy source.
*   `{DocumentSource.PHOTO.value}`: Captured via camera (phone, etc.).
*   `{DocumentSource.SCREENSHOT.value}`: Captured from a screen display.
*   `{DocumentSource.UNKNOWN.value}`: Source cannot be determined.
"""

    orientation_descriptions = f"""
*   `{Orientation.PORTRAIT.value}`: Standard vertical orientation.
*   `{Orientation.LANDSCAPE.value}`: Standard horizontal orientation.
*   `{Orientation.MIXED.value}`: Contains significant elements in both orientations (rare, use if truly mixed content).
*   `{Orientation.UNKNOWN.value}`: Orientation cannot be determined.
"""

    color_mode_descriptions = f"""
*   `{ColorMode.COLOR.value}`: Contains multiple colors beyond grayscale.
*   `{ColorMode.GRAYSCALE.value}`: Contains shades of gray, but no color.
*   `{ColorMode.BLACK_WHITE.value}`: Strictly two-tone (black and white pixels only).
*   `{ColorMode.UNKNOWN.value}`: Color mode cannot be determined.
"""

    language_descriptions = f"""
*   Select the dominant language using its ISO 639-1 code (e.g., `{LanguageCode.ENGLISH.value}`, `{LanguageCode.SPANISH.value}`, `{LanguageCode.FRENCH.value}`).
*   Use `{LanguageCode.MIXED.value}` if multiple languages have significant presence.
*   Use `{LanguageCode.OTHER.value}` if the language is not common or listed.
*   Use `{LanguageCode.UNKNOWN.value}` if the language cannot be identified.
*   MUST be one of the following exact values: {', '.join(f'`{v}`' for v in language_values)}
"""

    complexity_descriptions = f"""
*   `{Complexity.SIMPLE.value}`: Minimal variation in layout, few elements, straightforward text flow.
*   `{Complexity.MEDIUM.value}`: Moderate number of elements, standard layouts (e.g., single/double column with images), clear structure.
*   `{Complexity.COMPLEX.value}`: Multiple columns, complex tables, nested elements, varied formatting, significant non-text content.
*   `{Complexity.VERY_COMPLEX.value}`: Highly irregular layout, dense information, overlapping elements, unusual structures (e.g., complex forms, dense technical diagrams).
"""

    layout_style_descriptions = f"""
*   `{LayoutStyle.SINGLE_COLUMN.value}`: Text flows in one main column.
*   `{LayoutStyle.DOUBLE_COLUMN.value}`: Text flows in two distinct columns (common in papers, magazines).
*   `{LayoutStyle.MULTI_COLUMN.value}`: Text flows in three or more distinct columns (common in newspapers, some brochures).
*   `{LayoutStyle.COMPLEX.value}`: Combination of different column layouts, non-standard flow, or highly irregular structure.
*   `{LayoutStyle.IMAGE_HEAVY.value}`: Dominated by images/figures with relatively less text.
*   `{LayoutStyle.FORM_LIKE.value}`: Primarily structured as a form with fields and labels.
*   `{LayoutStyle.UNKNOWN.value}`: Layout style cannot be determined or doesn't fit categories.
"""

    segment_type_descriptions = f"""
*   `{SegmentType.TITLE.value}`: Main title of the document or a major section.
*   `{SegmentType.HEADING.value}`: Section or subsection heading, larger than body text.
*   `{SegmentType.SUBHEADING.value}`: Lower-level heading, smaller than Heading but distinct from body text.
*   `{SegmentType.CODE_BLOCK.value}`: Formatted block of source code.
*   `{SegmentType.LIST_ITEM.value}`: Individual item within a bulleted or numbered list.
*   `{SegmentType.BLOCK_QUOTE.value}`: Indented or specially formatted block of quoted text.
*   `{SegmentType.CAPTION.value}`: Text describing a figure, table, or image.
*   `{SegmentType.LEGEND.value}`: Key explaining symbols, colors, or patterns used in a visual element (e.g., chart, map, diagram). Distinct from a general caption.
*   `{SegmentType.FORMULA.value}`: Mathematical or chemical formula, often specially formatted or typeset.
*   `{SegmentType.HEADER.value}`: Repeating content at the top of the page (e.g., document title, chapter name).
*   `{SegmentType.FOOTER.value}`: Repeating content at the bottom of the page (e.g., confidentiality notice, document version).
*   `{SegmentType.PAGE_NUMBER.value}`: Number indicating the page sequence.
*   `{SegmentType.FOOTNOTE.value}`: Ancillary information placed at the bottom of the page or end of a section/document.
*   `{SegmentType.PICTURE.value}`: Visual elements like photographs, diagrams, charts, graphs.
*   `{SegmentType.TABLE.value}`: Data organized in rows and columns.
*   `{SegmentType.TEXT_BLOCK.value}`: Standard block of running text, the main content carrier.
*   `{SegmentType.FORM_REGION.value}`: Areas designed for data entry, often with labels and fields (e.g., key-value pairs).
*   `{SegmentType.SIGNATURE.value}`: Handwritten signature or designated area for one.
*   `{SegmentType.HANDWRITING.value}`: Significant portions of handwritten text (not just signatures).
*   `{SegmentType.GRAPHICAL_ITEM.value}`: Small graphical elements like logos, barcodes, QR codes, official stamps.
*   `{SegmentType.UNKNOWN.value}`: Segment type that doesn't fit into the above categories.
"""

    defect_descriptions = f"""
*   `{DefectType.SKEW.value}`: Page is rotated or slanted.
*   `{DefectType.BLUR.value}`: Text or image elements are out of focus or low resolution.
*   `{DefectType.WATERMARK.value}`: Overlay (text or image) partially obscuring content.
*   `{DefectType.STAIN.value}`: Discoloration from liquids, dirt, etc.
*   `{DefectType.CREASE.value}`: Visible fold lines on the page.
*   `{DefectType.HOLE_PUNCH.value}`: Holes from binder punching.
*   `{DefectType.LOW_CONTRAST.value}`: Text is faint or background interferes with readability.
*   `{DefectType.CUT_OFF.value}`: Edges of the document content are missing.
*   `{DefectType.SCANNED.value}`: Artifacts typical of scanning (e.g., scanner bed edges, slight noise, minor distortions not covered by skew/blur).
"""

    # --- Construct the main prompt ---
    prompt = f"""
Analyze the provided image, which is a single page from a document. Generate a JSON object that STRICTLY adheres to the provided JSON schema.

**JSON Output Structure and Field Instructions:**

1.  **`domain`**: (String Enum) Classify the primary subject matter based on the content AND overall appearance. Consider what kind of real-world document this page looks like it came from. Choose the BEST fit from the following options:
{domain_descriptions}
2.  **`source`**: (String Enum) Describe the origin or method of creation/capture. MUST be one of the following values:
{source_descriptions}
3.  **`orientation`**: (String Enum) Describe the intended orientation of the page content. MUST be one of the following values:
{orientation_descriptions}
4.  **`color_mode`**: (String Enum) Describe the color representation of the image. MUST be one of the following values:
{color_mode_descriptions}
5.  **`complexity`**: (String Enum) Estimate layout/content complexity. MUST be one of the following values. Choose based on these descriptions:
{complexity_descriptions}
6.  **`layout_style`**: (String Enum) Describe the overall page layout. MUST be one of the following values. Choose based on these descriptions:
{layout_style_descriptions}
7.  **`primary_language`**: (String Enum) Identify the main language.
{language_descriptions}
8.  **`segments`**: (List of Strings) List ALL distinct content segment types present on the page.
    *   This MUST be a JSON array (list) of strings.
    *   Each string in the list MUST be one of the following exact values:
{segment_type_descriptions}
    *   Include ONLY the segment types actually present on the page. If no segments of a certain type are found, DO NOT include that type in the list.
    *   **Example Format:** `["Heading", "Text Block/Paragraph", "Page Number"]`
9.  **`defects`**: (List of Strings) List any visual defects observed on the page.
    *   This MUST be a JSON array (list) of strings.
    *   Each string MUST be one of the following exact values:
{defect_descriptions}
    *   If no defects are observed, provide an empty list: `[]`.
10. **`is_sparse_or_empty`**: (Boolean) Set to `true` if the page has very little meaningful content (e.g., it's blank, nearly blank, a separator page, or contains ONLY elements like headers, footers, page numbers, or simple lines/borders). Set to `false` otherwise.

**Example JSON Output Structure (Do NOT include page_identifier):**

{{
  "domain": "research",
  "source": "{DocumentSource.SCANNED_CLEAN.value}",
  "orientation": "{Orientation.PORTRAIT.value}",
  "color_mode": "{ColorMode.BLACK_WHITE.value}",
  "complexity": "{Complexity.MEDIUM.value}",
  "layout_style": "{LayoutStyle.DOUBLE_COLUMN.value}",
  "primary_language": "{LanguageCode.ENGLISH.value}",
  "segments": ["{SegmentType.TEXT_BLOCK.value}", "{SegmentType.HEADING.value}", "{SegmentType.PICTURE.value}", "{SegmentType.FORMULA.value}", "{SegmentType.PAGE_NUMBER.value}", "{SegmentType.HEADER.value}"],
  "defects": ["{DefectType.SKEW.value}", "{DefectType.SCANNED.value}"],
  "is_sparse_or_empty": false
}}

**Output ONLY the JSON object, starting with `{{` and ending with `}}`. Do not include any explanatory text before or after the JSON.**
**Ensure all field names, types, and enum values match exactly based on the allowed values listed above.**
"""
    return prompt

def call_openrouter_llm(image_bytes: bytes, prompt: str, schema: Dict[str, Any], request_id_for_log: str) -> Optional[Dict[str, Any]]:
    """Sends image and prompt to OpenRouter multimodal model, enforcing schema."""
    if not OPENROUTER_API_KEY:
        print(f"  [{request_id_for_log}] Error: OPENROUTER_API_KEY is not set.")
        return None

    print(f"  [{request_id_for_log}] Encoding image and preparing LLM request...")
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Construct the response_format object using the provided Pydantic schema
        response_format_obj = {
            "type": "json_schema",
            "json_schema": {
                "name": "page_profile_extraction", # Descriptive name for the schema use
                "strict": True, # Enforce strict adherence
                "schema": schema # The actual JSON schema derived from Pydantic
            }
        }

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                # Ensure correct mime type if not always JPEG
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "response_format": response_format_obj, # Add the structured output format
            "max_tokens": 4096, # Adjust as needed, ensure it's enough for complex JSON
            "temperature": 1, # Keep low for deterministic JSON
        }

        print(f"  [{request_id_for_log}] Sending request to OpenRouter model: {OPENROUTER_MODEL}...")
        start_llm_call = time.time()
        response = requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=OPENROUTER_HEADERS,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS
        )
        llm_call_duration = time.time() - start_llm_call
        print(f"  [{request_id_for_log}] Received response from OpenRouter in {llm_call_duration:.2f}s. Status: {response.status_code}")

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        result = response.json()

        # Extract the JSON content from the response
        if result.get("choices") and len(result["choices"]) > 0:
            message = result["choices"][0].get("message")
            if message and message.get("content"):
                content = message["content"]
                # OpenRouter with json_schema mode should return JSON directly
                # No need to strip markdown backticks usually, but handle if necessary
                if isinstance(content, str):
                    try:
                        # Attempt to parse if it's a string (fallback)
                        extracted_json = json.loads(content)
                        print(f"  [{request_id_for_log}] Parsed JSON string response.")
                    except json.JSONDecodeError as json_err:
                        print(f"  [{request_id_for_log}] Error: LLM response content is a string but not valid JSON: {json_err}")
                        print(f"  Raw content: {content[:500]}...") # Log snippet
                        return None
                elif isinstance(content, dict):
                    # Ideal case: content is already a dictionary
                    extracted_json = content
                    print(f"  [{request_id_for_log}] Received structured JSON response.")
                else:
                    print(f"  [{request_id_for_log}] Error: Unexpected content type in LLM response: {type(content)}")
                    return None

                return extracted_json
            else:
                 print(f"  [{request_id_for_log}] Error: LLM response missing 'message' or 'content'.")
                 return None
        else:
            print(f"  [{request_id_for_log}] Error: LLM response missing 'choices' or choices list is empty.")
            # Log the raw response for debugging
            print(f"  Raw response data: {result}")
            return None

    except requests.exceptions.Timeout:
        print(f"  [{request_id_for_log}] Error: Timeout ({LLM_TIMEOUT_SECONDS}s) calling OpenRouter API.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [{request_id_for_log}] Error: Network or HTTP error calling OpenRouter API: {e}")
        # Log response body if available and useful
        if e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            try:
                print(f"  Response body: {e.response.text[:500]}...") # Log snippet
            except Exception:
                print("  Could not read response body.")
        return None
    except json.JSONDecodeError as e:
        print(f"  [{request_id_for_log}] Error: Failed to decode JSON response from OpenRouter: {e}")
        # Log the raw response text if possible
        try:
            print(f"  Raw response text: {response.text[:500]}...") # Log snippet
        except Exception:
             print("  Could not get raw response text.")
        return None
    except Exception as e:
        print(f"  [{request_id_for_log}] Unexpected error during LLM call or processing: {e}")
        # Consider logging traceback here for complex errors
        # import traceback
        # traceback.print_exc()
        return None

def upload_json_to_aws(client, bucket_name: str, key: str, data: Dict[str, Any]):
    """Uploads the extracted JSON data to AWS S3."""
    try:
        client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json.dumps(data, indent=2), # Pretty print JSON
            ContentType='application/json'
        )
        # print(f"  Uploaded JSON to AWS S3: s3://{bucket_name}/{key}") # Verbose log
    except ClientError as e:
        print(f"  Error uploading JSON to AWS S3 key {key}: {e}")
        raise # Re-raise to signal failure in the worker
    except Exception as e:
        print(f"  Unexpected error uploading JSON to AWS S3 key {key}: {e}")
        raise # Re-raise to signal failure in the worker

# --- Failed DB Handling ---
def initialize_failed_db(db_path: Path):
    """Creates the failed_pages table if it doesn't exist."""
    print(f"Initializing failed extractions database at: {db_path}")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_pages (
                    dedup_id TEXT PRIMARY KEY,
                    gcs_key TEXT NOT NULL,
                    failure_reason TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_failed_timestamp ON failed_pages (timestamp);")
        print(f"Failed extractions database initialized successfully.")
    except sqlite3.Error as e:
        print(f"ERROR: Could not initialize failed extractions database at {db_path}: {e}")
        raise # Re-raise the exception to halt execution if DB init fails

def record_failed_page(conn: sqlite3.Connection, dedup_id: str, gcs_key: str, reason: str):
    """Records a failed page into the failed_extractions database."""
    try:
        conn.execute("""
            INSERT OR REPLACE INTO failed_pages (dedup_id, gcs_key, failure_reason)
            VALUES (?, ?, ?);
        """, (dedup_id, gcs_key, reason))
        conn.commit() # Commit immediately after insert
    except sqlite3.Error as e:
        print(f"ERROR: Could not record failed page {dedup_id} (GCS: {gcs_key}) to database: {e}")

# <<< Add function to remove successfully rerun pages >>>
def remove_successful_rerun(conn: sqlite3.Connection, dedup_id: str):
    """Removes a successfully rerun page from the failed_pages table."""
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM failed_pages WHERE dedup_id = ?;", (dedup_id,))
        conn.commit()
        if cursor.rowcount > 0:
            print(f"[{dedup_id}] Successfully rerun page removed from failed DB.")
        # else: # Optional: Log if ID wasn't found (shouldn't happen in normal flow)
        #     print(f"[{dedup_id}] Warning: Tried to remove from failed DB, but ID not found.")
    except sqlite3.Error as e:
        print(f"ERROR: Could not remove page {dedup_id} from failed DB: {e}")

# --- Worker Function ---
def process_page_worker(
    page_dedup_id: str,
    gcs_key: str,
    gcs_client,
    gcs_bucket: str,
    aws_client,
    aws_bucket: str,
    page_schema: Dict[str, Any],
    skip_validation: bool
) -> Tuple[str, bool, str]:
    """
    Worker task: download, analyze via LLM, optionally validate, upload.
    """
    print(f"[{page_dedup_id}] Starting processing for GCS key: {gcs_key}")

    # 1. Download from GCS
    print(f"[{page_dedup_id}] Downloading from GCS...")
    image_bytes = download_page_from_gcs(gcs_client, gcs_bucket, gcs_key)
    if image_bytes is None:
        return page_dedup_id, False, "GCS Download Failed"

    # 2. Generate Prompt
    prompt = generate_llm_prompt()

    # 3. Call LLM for extraction
    print(f"[{page_dedup_id}] Calling LLM...")
    extracted_data = call_openrouter_llm(image_bytes, prompt, page_schema, page_dedup_id)
    if extracted_data is None:
        return page_dedup_id, False, "LLM Call Failed/Timeout/Error"

    # 4. Validate with Pydantic (Conditional)
    page_domain = "unknown_domain" # Default domain
    upload_data = None # Initialize upload_data

    # Log the raw data before validation attempt for debugging
    if not skip_validation:
        print(f"[{page_dedup_id}] Raw data for validation: {extracted_data}")

    if skip_validation:
        # --- VALIDATION BYPASSED ---
        print(f"[{page_dedup_id}] Skipping Pydantic validation as requested (--skip-validation flag set).")
        upload_data = extracted_data # Use raw data

        # Attempt to get domain safely for S3 path
        if isinstance(extracted_data, dict) and isinstance(extracted_data.get("domain"), str):
            page_domain = extracted_data.get("domain", "unknown_domain")
            page_domain = page_domain.replace("/", "_").replace(" ", "_").lower()
        elif isinstance(extracted_data, dict):
            page_domain = f"invalid_domain_type_{type(extracted_data.get('domain')).__name__}"
        # If not a dict, page_domain remains "unknown_domain"

    else:
        # --- PERFORM VALIDATION ---
        try:
            validated_profile = PageProfile.model_validate(extracted_data)
            print(f"[{page_dedup_id}] LLM Response validated successfully against schema.")
            upload_data = validated_profile.model_dump() # Use validated & dumped data
            page_domain = validated_profile.domain.value # Get domain from validated data

        except ValidationError as e:
            print(f"[{page_dedup_id}] Error: LLM response failed Pydantic validation!")
            # Print the detailed validation error to understand the exact failure
            print(f"[{page_dedup_id}] Validation Error Details:\n{e}\n")
            # Return failure specifically for validation error
            return page_dedup_id, False, "LLM Validation Failed"
        except Exception as e:
            # Catch other potential errors during validation/dumping
            print(f"[{page_dedup_id}] Error during validation/data handling: {e}")
            return page_dedup_id, False, f"Validation/Data Handling Error: {e}"

    # Ensure upload_data is assigned (should be unless validation failed)
    if upload_data is None:
         # This case should ideally not be reached if validation is skipped or succeeds
         print(f"[{page_dedup_id}] Error: upload_data is None before upload attempt. This shouldn't happen.")
         return page_dedup_id, False, "Internal Worker Error (upload_data None)"

    # Add page_identifier programmatically (applies to both validated and raw data)
    if isinstance(upload_data, dict):
         upload_data['page_identifier'] = page_dedup_id
    else:
         # Handle cases where LLM returned non-dict and validation was skipped
         print(f"[{page_dedup_id}] Warning: LLM response was not a dictionary (type: {type(upload_data)}) and validation was skipped. Wrapping response.")
         upload_data = {
             "page_identifier": page_dedup_id,
             "raw_llm_response": upload_data # Store the original non-dict response
         }

    # 5. Upload JSON and Image to AWS S3
    aws_object_key = f"dla-dataset/{page_domain}/{page_dedup_id}/extracted_data.json"
    image_s3_key = f"dla-dataset/{page_domain}/{page_dedup_id}/source_image.jpg"
    status_suffix = "Validation Bypassed" if skip_validation else "Validated"

    try:
        # Upload JSON
        print(f"[{page_dedup_id}] Uploading ({status_suffix}) JSON to AWS S3: s3://{aws_bucket}/{aws_object_key}")
        upload_json_to_aws(aws_client, aws_bucket, aws_object_key, upload_data)

        # Upload Image
        print(f"[{page_dedup_id}] Uploading Image to AWS S3: s3://{aws_bucket}/{image_s3_key}")
        aws_client.put_object(
            Bucket=aws_bucket,
            Key=image_s3_key,
            Body=image_bytes, # The image data downloaded from GCS
            ContentType='image/jpeg' # Assuming JPEG, adjust if needed
        )

        # Update success message to reflect both uploads
        return page_dedup_id, True, f"Success ({status_suffix}, Domain: {page_domain}, JSON+Image Uploaded)"
    except ClientError as e: # Catch specific S3 errors if possible
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        print(f"[{page_dedup_id}] Error uploading to AWS S3: {error_code}")
        # Consider more specific error handling if needed
        return page_dedup_id, False, f"AWS S3 Upload Failed ({error_code})"
    except Exception as e: # Catch other potential errors during upload
        print(f"[{page_dedup_id}] Unexpected error during AWS upload: {e}")
        return page_dedup_id, False, f"AWS Upload Failed (Unexpected: {e})"

# --- Main Execution ---
def run_extraction(rerun_failed: bool, skip_validation: bool):
    """Main function to run the extraction pipeline."""
    print("-" * 40)
    run_mode = "RERUN FAILED" if rerun_failed else "NORMAL"
    validation_mode = "Validation SKIPPED" if skip_validation else "Validation ENABLED"
    print(f"Starting Extraction Pipeline in {run_mode} mode.")
    print(f"Schema Validation: {validation_mode}")
    print("-" * 40)

    start_time = time.time()
    processed_count = 0
    count_success = 0
    count_validation_failed = 0 # Track validation fails again
    count_error = 0 # Other errors (GCS, AWS, LLM Call, Worker crash)
    successfully_rerun_ids = set()
    last_print_time = time.time() # Initialize last_print_time here

    # --- Setup Clients and Schema ---
    try:
        gcs_settings = get_gcs_settings()
        gcs_client = get_gcs_client(gcs_settings)
        aws_client = get_aws_client()
        aws_bucket = AWS_BUCKET_NAME
        page_profile_schema_dict = PageProfile.model_json_schema()
    except ValueError as e:
         print(f"Configuration Error: {e}")
         return # Exit if essential config is missing
    except Exception as e:
        print(f"Unexpected error during setup: {e}")
        return

    # --- Initialize Failed DB Connection ---
    failed_db_conn = None
    try:
        # Initialize DB (creates if not exists) - needed for both modes
        initialize_failed_db(FAILED_DB_PATH)
        # Connect for recording failures (and potentially deleting in rerun mode)
        failed_db_conn = sqlite3.connect(FAILED_DB_PATH)
    except Exception as e:
        print(f"ERROR: Failed to initialize or connect to {FAILED_DB_PATH.name}: {e}")
        # Decide if you want to continue without failed DB logging
        print("Warning: Proceeding without failed DB logging/updates.")
        failed_db_conn = None # Ensure it's None if connection failed

    # --- Get Pages to Process ---
    pages_to_process: List[Tuple[str, str]] = []
    try:
        if rerun_failed:
            pages_to_process = get_failed_pages_from_db(FAILED_DB_PATH)
            if not pages_to_process:
                print("Exiting as there are no failed pages to rerun.")
                if failed_db_conn: failed_db_conn.close()
                return # Exit early
        else:
            pages_to_process = get_unique_pages_from_db(DB_PATH)
            if not pages_to_process:
                print("Exiting as no unique pages were found in the main database.")
                if failed_db_conn: failed_db_conn.close()
                return # Exit early

    except FileNotFoundError as e:
        print(f"Error: {e}")
        if failed_db_conn: failed_db_conn.close()
        return
    except Exception as e: # Catch other potential DB errors during fetch
        print(f"An unexpected error occurred fetching pages: {e}")
        if failed_db_conn: failed_db_conn.close()
        return

    total_pages = len(pages_to_process)
    print(f"Total pages to process: {total_pages}")

    # --- ThreadPoolExecutor ---
    future_to_info: Dict[Any, Tuple[str, str]] = {}
    print_interval_seconds = 30 # Print progress interval

    try:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit initial batch of tasks
            submitted_count = 0
            for dedup_id, gcs_key in pages_to_process:
                future = executor.submit(
                    process_page_worker,
                    dedup_id, gcs_key, gcs_client, gcs_settings.bucket_name,
                    aws_client, aws_bucket, page_profile_schema_dict,
                    skip_validation # <<< Pass the flag value to the worker
                )
                future_to_info[future] = (dedup_id, gcs_key)
                submitted_count += 1
                # time.sleep(0.01) # Optional throttling

            print(f"Submitted all {submitted_count} tasks to the executor.")

            # Process completed tasks
            for future in as_completed(future_to_info):
                processed_count += 1
                original_dedup_id, original_gcs_key = future_to_info[future]
                try:
                    page_id_result, success, status_msg = future.result()

                    if success:
                        count_success += 1
                        print(f"[{page_id_result}] Success. Status: {status_msg}")
                        # If in rerun mode, mark this ID for removal from failed DB
                        if rerun_failed:
                            successfully_rerun_ids.add(page_id_result)
                    else:
                        # Check if it was a validation failure specifically
                        if status_msg == "LLM Validation Failed":
                            count_validation_failed += 1
                            print(f"[{page_id_result}] Failed. Status: {status_msg}. Recording to failed DB.")
                            # Record the failure only if validation was attempted
                            # And only if not in rerun mode (or if you want to update)
                            if not rerun_failed and failed_db_conn:
                                 record_failed_page(failed_db_conn, original_dedup_id, original_gcs_key, status_msg)
                            elif rerun_failed:
                                print(f"[{page_id_result}] Note: Keeping entry in failed DB as rerun failed validation again.")
                            else:
                                 print(f"[{page_id_result}] Warning: Cannot record failure, failed DB connection is not available.")
                        else:
                            # Other types of failures (GCS, AWS, LLM Call, Worker Error)
                            count_error += 1
                            print(f"[{page_id_result}] Failed. Status: {status_msg}")
                            # <<< UNCOMMENT THE FOLLOWING LINES >>>
                            if failed_db_conn:
                                # Log these errors regardless of rerun mode, or add 'if not rerun_failed:' if desired
                                record_failed_page(failed_db_conn, original_dedup_id, original_gcs_key, f"Worker Error: {status_msg}")

                except Exception as exc:
                    # Handle exceptions raised by the worker task itself
                    count_error += 1
                    print(f"[{original_dedup_id}] Error processing future: {exc}")
                    # <<< UNCOMMENT THE FOLLOWING LINES >>>
                    if failed_db_conn:
                        # Log these errors regardless of rerun mode, or add 'if not rerun_failed:' if desired
                        record_failed_page(failed_db_conn, original_dedup_id, original_gcs_key, f"Future Error: {exc}")

                # --- Progress Update ---
                current_time = time.time()
                if current_time - last_print_time >= print_interval_seconds or processed_count == total_pages:
                    elapsed_time = current_time - start_time
                    pages_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                    est_total_time = (elapsed_time / processed_count * total_pages) if processed_count > 0 else 0
                    remaining_time = est_total_time - elapsed_time if est_total_time > elapsed_time else 0
                    time_remaining_str = time.strftime('%H:%M:%S', time.gmtime(remaining_time)) if remaining_time > 0 else "N/A"

                    print(f"\n--- Progress ({processed_count}/{total_pages}) ---")
                    print(f"  Success: {count_success}")
                    # Only show validation failed count if validation is enabled
                    if not skip_validation:
                        print(f"  Validation Failed: {count_validation_failed}")
                    print(f"  Other Errors: {count_error}")
                    print(f"  Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                    print(f"  Current Rate: {pages_per_second:.2f} pages/sec")
                    print(f"  Est. Time Remaining: {time_remaining_str}")
                    print(f"-----------------------\n")
                    last_print_time = current_time

    except Exception as e:
        print(f"An unexpected error occurred during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Remove successfully rerun items from failed DB ---
        if rerun_failed and successfully_rerun_ids and failed_db_conn:
            print(f"Removing {len(successfully_rerun_ids)} successfully rerun pages from {FAILED_DB_PATH.name}...")
            removed_count = 0
            for dedup_id in successfully_rerun_ids:
                try:
                    remove_successful_rerun(failed_db_conn, dedup_id)
                    removed_count +=1
                except Exception as e:
                    print(f"Error removing {dedup_id} from failed DB: {e}") # Log error but continue
            print(f"Finished removing {removed_count} pages from failed DB.")

        # --- Final Summary ---
        if failed_db_conn:
            print("Closing failed extractions database connection.")
            failed_db_conn.close()

        end_time = time.time()
        total_elapsed = end_time - start_time
        final_rate = processed_count / total_elapsed if total_elapsed > 0 else 0
        print("-" * 40)
        mode = "Rerun" if rerun_failed else "Extraction"
        validation_status = "(Validation Bypassed)" if skip_validation else ""
        print(f"{mode} Pipeline Finished {validation_status}.")
        print(f"  Total pages processed attempts: {processed_count}")
        print(f"  Successful uploads: {count_success}")
        if rerun_failed:
             print(f"  Successfully processed & removed from failed DB: {len(successfully_rerun_ids)}")
        # Only show validation failed count if validation is enabled
        if not skip_validation:
            db_log_target = FAILED_DB_PATH.name if not rerun_failed else "(Not re-logged in rerun)"
            print(f"  LLM Validation Failures (logged to {db_log_target}): {count_validation_failed}")
        print(f"  Other Errors encountered (GCS/AWS/LLM Call/Worker): {count_error}")
        print(f"  Total time elapsed: {time.strftime('%H:%M:%S', time.gmtime(total_elapsed))}")
        print(f"  Average rate: {final_rate:.2f} pages/sec")
        print("-" * 40)


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the DLA page extraction pipeline.")
    parser.add_argument(
        '--rerun-failed',
        action='store_true',
        help='If set, process only pages listed in failed_extractions.db.'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='If set, bypass Pydantic validation and upload raw LLM output.'
    )
    args = parser.parse_args()

    # --- Run Main Function ---
    run_extraction(rerun_failed=args.rerun_failed, skip_validation=args.skip_validation)