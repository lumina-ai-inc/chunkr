import streamlit as st
import psycopg
from psycopg import rows
import boto3
from botocore.exceptions import ClientError
import os
import json
import random
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
from psycopg import sql

# Import Enums from your schema.py
# Assuming schema.py is in the same directory or accessible in PYTHONPATH
try:
    from schema import (
        DocumentSource, Orientation, ColorMode, Complexity, LayoutStyle,
        LanguageCode, SegmentType, DefectType
    )
except ImportError:
    st.error("Failed to import Enums from schema.py. Make sure it's in the correct path.")
    # Fallback to empty lists or handle error appropriately if Enums are critical for startup
    DocumentSource, Orientation, ColorMode, Complexity, LayoutStyle = [], [], [], [], []
    LanguageCode, SegmentType, DefectType = [], [], []


# --- Environment Loading (mimicking upload_to_postgres.py) ---
# Load environment variables from .env file located three levels up
# (assuming .env is in machine-learning/datasets/dla/)
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
print(f"Attempting to load .env file for DLA Explorer from: {dotenv_path}")
if dotenv_path.is_file():
    print(f".env file found at {dotenv_path}. Loading variables...")
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    print(f"WARNING: .env file not found at {dotenv_path}. Relying on pre-set environment variables.")


# --- Configuration ---
# PostgreSQL Configuration
DATABASE_URL = os.environ.get("DATABASE_URL")

# Fallback individual components (primarily for the error message if DATABASE_URL is missing)
PG_USER_FALLBACK = os.environ.get("POSTGRES_USER")
PG_PASSWORD_FALLBACK = os.environ.get("POSTGRES_PASSWORD")
PG_DBNAME_FALLBACK = os.environ.get("POSTGRES_DB_NAME")

# AWS S3 Configuration - using names from your .env
AWS_ACCESS_KEY_ID = os.environ.get("AWS__ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS__SECRET_KEY")
AWS_S3_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
AWS_S3_ENDPOINT_URL = os.environ.get("AWS__ENDPOINT")
AWS_REGION_NAME = os.environ.get("AWS_REGION_NAME") # Optional

TAGGING_SET_TABLE_NAME = "dla_selected_tagging_pages" # New global constant

# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    conn_string = DATABASE_URL
    if conn_string:
        # print("DEBUG: Attempting DB connection using DATABASE_URL.") # Optional debug
        pass
    else:
        st.error(
            "PostgreSQL DATABASE_URL is not configured. "
            "Please set the DATABASE_URL environment variable. "
            "Ensure your .env file is correctly loaded or variables are pre-set."
        )
        return None
    try:
        conn = psycopg.connect(conn_string)
        # print("DEBUG: DB Connection Successful.") # Optional debug
        return conn
    except psycopg.Error as e:
        st.error(f"Error connecting to PostgreSQL using DATABASE_URL: {e}")
        return None

# --- S3 Client ---
@st.cache_resource
def get_s3_client():
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET_NAME]):
        st.error(
            "AWS S3 credentials or bucket name not fully configured. "
            "Please ensure AWS__ACCESS_KEY, AWS__SECRET_KEY, and AWS_BUCKET_NAME "
            "environment variables are set correctly (e.g., via .env file or pre-set)."
        )
        return None
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION_NAME,
            endpoint_url=AWS_S3_ENDPOINT_URL
        )
        # print("DEBUG: S3 Client Created.") # Optional debug
        return client
    except ClientError as e:
        st.error(f"Error creating S3 client: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error creating S3 client: {e}")
        return None

# --- Helper function to check table existence ---
@st.cache_data(ttl=60) # Cache for a short period
def check_table_exists(_conn: psycopg.Connection, table_name: str) -> bool:
    if not _conn:
        return False
    try:
        with _conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                );
            """, (table_name,))
            exists = cur.fetchone()
            return exists[0] if exists else False
    except psycopg.Error as e:
        st.error(f"Error checking if table '{table_name}' exists: {e}")
        return False

# --- Data Fetching Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_domains(_conn: psycopg.Connection, show_tagging_set_only: bool = False, tagging_set_table_exists: bool = False) -> List[str]:
    if not _conn:
        return []
    try:
        with _conn.cursor() as cur:
            query_parts: List[sql.Composable] = [
                sql.SQL("SELECT DISTINCT pp.domain FROM {profiles} pp").format(profiles=sql.Identifier('page_profiles'))
            ]
            if show_tagging_set_only and tagging_set_table_exists:
                query_parts.append(
                    sql.SQL("JOIN {tag_tbl} tsp ON pp.page_identifier = tsp.page_identifier").format(
                        tag_tbl=sql.Identifier(TAGGING_SET_TABLE_NAME)
                    )
                )
            query_parts.append(sql.SQL("ORDER BY pp.domain"))
            
            final_query = sql.SQL(' ').join(query_parts)
            cur.execute(final_query)
            domains = [row[0] for row in cur.fetchall()]
            return domains
    except psycopg.Error as e:
        st.error(f"Error fetching domains: {e}")
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_pages_for_domain(
    _conn: psycopg.Connection,
    domain: str | None,
    sources: Optional[List[str]] = None,
    orientations: Optional[List[str]] = None,
    color_modes: Optional[List[str]] = None,
    complexities: Optional[List[str]] = None,
    layout_styles: Optional[List[str]] = None,
    primary_languages: Optional[List[str]] = None,
    sparse_empty_status: Optional[bool] = None,
    segments_filter: Optional[List[str]] = None,
    defects_filter: Optional[List[str]] = None,
    show_tagging_set_only: bool = False,       # New parameter
    tagging_set_table_exists: bool = False   # New parameter
) -> List[Dict[str, str]]:
    if not _conn:
        return []

    from psycopg import sql

    # 1) Build the SELECT list, always qualify with pp alias
    fields = sql.SQL(', ').join([
        sql.Identifier('pp', 'page_identifier'),
        sql.Identifier('pp', 'aws_s3_image_key'),
        sql.Identifier('pp', 'aws_s3_json_key'),
    ])

    # 2) Start composing the query
    query = sql.SQL("SELECT DISTINCT {fields} FROM {profiles} pp").format(
        fields=fields,
        profiles=sql.Identifier('page_profiles'),
    )

    params: List[Any] = []
    where_clauses: List[sql.Composable] = []
    join_clauses: List[sql.Composable] = []

    # Join for tagging set if requested and table exists
    if show_tagging_set_only and tagging_set_table_exists:
        join_clauses.append(
            sql.SQL("JOIN {tag_tbl} tsp ON pp.page_identifier = tsp.page_identifier").format(
                tag_tbl=sql.Identifier(TAGGING_SET_TABLE_NAME)
            )
        )

    if domain:
        where_clauses.append(
            sql.SQL("pp.domain = {}").format(sql.Placeholder())
        )
        params.append(domain)

    # 3) Optional simple filters → WHERE ... = ANY(%s)
    def _add_any_filter(col: str, values: Optional[List[str]]):
        if values:
            where_clauses.append(
                sql.SQL("pp.{col} = ANY({ph})").format(
                    col=sql.Identifier(col),
                    ph=sql.Placeholder()
                )
            )
            params.append(values)

    _add_any_filter('source', sources)
    _add_any_filter('orientation', orientations)
    _add_any_filter('color_mode', color_modes)
    _add_any_filter('complexity', complexities)
    _add_any_filter('layout_style', layout_styles)
    _add_any_filter('primary_language', primary_languages)

    if sparse_empty_status is not None:
        where_clauses.append(
            sql.SQL("pp.is_sparse_or_empty = {}").format(sql.Placeholder())
        )
        params.append(sparse_empty_status)

    # 4) JOINs for segments & defects (at least one match)
    if segments_filter:
        join_clauses.append(
            sql.SQL(
                "JOIN {seg_tbl} ps ON pp.page_identifier = ps.page_identifier "
                "AND ps.segment_type = ANY({ph})"
            ).format(
                seg_tbl=sql.Identifier('page_segments'),
                ph=sql.Placeholder()
            )
        )
        params.append(segments_filter)

    if defects_filter:
        join_clauses.append(
            sql.SQL(
                "JOIN {def_tbl} pd ON pp.page_identifier = pd.page_identifier "
                "AND pd.defect_type = ANY({ph})"
            ).format(
                def_tbl=sql.Identifier('page_defects'),
                ph=sql.Placeholder()
            )
        )
        params.append(defects_filter)

    # 5) Stitch it all together
    if join_clauses:
        query = query + sql.SQL(" ") + sql.SQL(" ").join(join_clauses)

    if where_clauses:
        query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)

    query = query + sql.SQL(" ORDER BY ") + sql.Identifier('pp', 'page_identifier')

    # 6) Execute prepared query
    try:
        with _conn.cursor(row_factory=rows.dict_row) as cur:
            cur.execute(query, params)
            return cur.fetchall()
    except psycopg.Error as e:
        # if one query fails, roll back so the conn isn't "stuck" in an aborted state
        try:
            _conn.rollback()
        except Exception:
            pass
        st.error(f"Error fetching pages for domain '{domain}' with filters: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_page_details(_conn: psycopg.Connection, page_identifier: str) -> Optional[Dict[str, Any]]:
    if not _conn or not page_identifier:
        return None
    try:
        with _conn.cursor(row_factory=rows.dict_row) as cur:
            # Fetch main profile
            cur.execute(
                "SELECT * FROM page_profiles WHERE page_identifier = %s;",
                (page_identifier,)
            )
            details = cur.fetchone()
            if not details:
                return None

            # Fetch segments
            cur.execute(
                "SELECT segment_type FROM page_segments WHERE page_identifier = %s;",
                (page_identifier,)
            )
            details['segments'] = [row['segment_type'] for row in cur.fetchall()]

            # Fetch defects
            cur.execute(
                "SELECT defect_type FROM page_defects WHERE page_identifier = %s;",
                (page_identifier,)
            )
            details['defects'] = [row['defect_type'] for row in cur.fetchall()]
            return details
    except psycopg.Error as e:
        st.error(f"Error fetching details for page '{page_identifier}': {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_s3_object(_s3_client, bucket_name: str | None, key: str | None, is_json: bool = False) -> Optional[Any]:
    if not _s3_client or not bucket_name or not key:
        return None
    try:
        response = _s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read()
        if is_json:
            return json.loads(content.decode('utf-8'))
        else:
            return BytesIO(content) # For images
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            st.warning(f"S3 Error: Key not found - s3://{bucket_name}/{key}")
        else:
            st.error(f"S3 ClientError downloading {key}: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"JSON Error: Failed to parse {key}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error downloading/parsing {key}: {e}")
        return None

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="DLA Explorer")
st.title("📄 DLA Page Explorer")

conn = get_db_connection()
s3_client = get_s3_client()

if not conn or not s3_client:
    st.warning("Application cannot start due to missing DB or S3 configuration. Please check logs and environment variables.")
    st.stop()

# Initialize session state variables
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = "All"
if 'pages_in_domain' not in st.session_state:
    st.session_state.pages_in_domain = []
if 'current_page_index' not in st.session_state:
    st.session_state.current_page_index = 0
if 'shuffled_indices' not in st.session_state: # To store the shuffled order of page indices
    st.session_state.shuffled_indices = []
if 'force_page_reset' not in st.session_state:
    st.session_state.force_page_reset = False

# Initialize filter session states
filter_keys_defaults = {
    "filters_source": [], "filters_orientation": [], "filters_color_mode": [],
    "filters_complexity": [], "filters_layout_style": [], "filters_language": [],
    "filter_sparse_value_display": "Any", # For selectbox display
    "filters_segments": [], "filters_defects": []
}
for key, default in filter_keys_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Initialize session state for the checkbox if not present
if 'cb_show_tagging_set' not in st.session_state:
    st.session_state.cb_show_tagging_set = False

def tagging_set_toggle_changed():
    st.session_state.force_page_reset = True

# --- Sidebar for Domain Selection ---
st.sidebar.header("Select Domain")

# --- Dataset View Toggle ---
st.sidebar.header("Dataset View")
show_tagging_set_checkbox = st.sidebar.checkbox(
    "Show only Tagging Set Pages",
    key='cb_show_tagging_set',
    on_change=tagging_set_toggle_changed,
    help=f"Filters pages to include only those in the '{TAGGING_SET_TABLE_NAME}' table (if it exists)."
)

# Check if tagging set table exists (do this after connection is established)
tagging_table_actually_exists = False
if conn: # Ensure connection exists before checking table
    tagging_table_actually_exists = check_table_exists(conn, TAGGING_SET_TABLE_NAME)

if show_tagging_set_checkbox and not tagging_table_actually_exists:
    st.sidebar.warning(f"'{TAGGING_SET_TABLE_NAME}' table not found. Displaying all pages.")

# Determine effective filtering status for tagging set
should_filter_for_tagging_set = show_tagging_set_checkbox and tagging_table_actually_exists

available_domains = fetch_domains(conn, show_tagging_set_only=should_filter_for_tagging_set, tagging_set_table_exists=tagging_table_actually_exists)
# prepend an "All" option and pick its index
domains_list = ["All"] + available_domains

# Ensure selected_domain is valid after potential list change
if st.session_state.selected_domain not in domains_list:
    st.session_state.selected_domain = "All" # Default to "All" if previous selection is no longer valid
    st.session_state.force_page_reset = True # Force reload if domain was reset

current_domain_index = domains_list.index(st.session_state.selected_domain)

selected_domain_name = st.sidebar.selectbox(
    "Choose a domain:",
    domains_list,
    index=current_domain_index,
    key="domain_selector"
)

domain_changed = False
if selected_domain_name != st.session_state.selected_domain:
    st.session_state.selected_domain = selected_domain_name
    domain_changed = True
    # Reset all other filters and page state when domain changes
    for key, default_value in filter_keys_defaults.items():
        st.session_state[key] = default_value
    st.session_state.force_page_reset = True

# Ensure pages_in_domain is populated on first load, domain change, or filter change
reload_needed = st.session_state.force_page_reset or not st.session_state.pages_in_domain
if st.session_state.selected_domain and reload_needed:
    # Map the "Any"/"Yes"/"No" display into actual None/True/False
    sparse_options_map = {"Any": None, "Yes": True, "No": False}
    actual_sparse = sparse_options_map.get(
        st.session_state.filter_sparse_value_display,
        None
    )
    # treat "All" as None → no domain filter
    domain_param = None if st.session_state.selected_domain == "All" else st.session_state.selected_domain
    st.session_state.pages_in_domain = fetch_pages_for_domain(
        conn,
        domain_param,
        sources=st.session_state.filters_source,
        orientations=st.session_state.filters_orientation,
        color_modes=st.session_state.filters_color_mode,
        complexities=st.session_state.filters_complexity,
        layout_styles=st.session_state.filters_layout_style,
        primary_languages=st.session_state.filters_language,
        sparse_empty_status=actual_sparse,
        segments_filter=st.session_state.filters_segments,
        defects_filter=st.session_state.filters_defects,
        show_tagging_set_only=should_filter_for_tagging_set, # Pass the effective filter status
        tagging_set_table_exists=tagging_table_actually_exists # Pass table existence status
    )
    # reset index & shuffle, then clear the reload flag
    st.session_state.current_page_index = 0
    st.session_state.shuffled_indices = list(range(len(st.session_state.pages_in_domain)))
    st.session_state.force_page_reset = False

# --- Filters ---
st.sidebar.header("Filters")

# Helper for multiselect filters
def create_filter_multiselect(label: str, enum_class: Any, session_key: str):
    if not enum_class: # Handle case where enum failed to import
        st.sidebar.warning(f"Options for '{label}' unavailable.")
        return
    options = [e.value for e in enum_class]
    current_selection = st.session_state.get(session_key, [])
    new_selection = st.sidebar.multiselect(label, options, default=current_selection, key=f"ms_{session_key}")
    if new_selection != current_selection:
        st.session_state[session_key] = new_selection
        st.session_state.force_page_reset = True # Mark that filters changed

create_filter_multiselect("Source:", DocumentSource, "filters_source")
create_filter_multiselect("Orientation:", Orientation, "filters_orientation")
create_filter_multiselect("Color Mode:", ColorMode, "filters_color_mode")
create_filter_multiselect("Complexity:", Complexity, "filters_complexity")
create_filter_multiselect("Layout Style:", LayoutStyle, "filters_layout_style")
create_filter_multiselect("Primary Language:", LanguageCode, "filters_language")

# is_sparse_or_empty filter
sparse_options_map = {"Any": None, "Yes": True, "No": False}
current_sparse_display = st.session_state.filter_sparse_value_display
new_sparse_display = st.sidebar.selectbox(
    "Is Sparse/Empty:",
    options=list(sparse_options_map.keys()),
    index=list(sparse_options_map.keys()).index(current_sparse_display),
    key="sb_sparse_empty"
)
if new_sparse_display != current_sparse_display:
    st.session_state.filter_sparse_value_display = new_sparse_display
    st.session_state.force_page_reset = True

create_filter_multiselect("Segments (page has any of):", SegmentType, "filters_segments")
create_filter_multiselect("Defects (page has any of):", DefectType, "filters_defects")


# --- Fetching Pages based on Domain and Filters ---
# This section runs after sidebar is built and session_state is updated by widgets

# Convert display value of sparse filter to actual boolean/None
actual_sparse_filter = sparse_options_map[st.session_state.filter_sparse_value_display]

# Fetch or re-fetch pages if domain changed, filters changed, or pages not yet loaded
if domain_changed or st.session_state.force_page_reset or not st.session_state.pages_in_domain:
    # treat "All" as None → no domain filter
    domain_param = None if st.session_state.selected_domain == "All" else st.session_state.selected_domain
    st.session_state.pages_in_domain = fetch_pages_for_domain(
        conn,
        domain_param,
        sources=st.session_state.filters_source,
        orientations=st.session_state.filters_orientation,
        color_modes=st.session_state.filters_color_mode,
        complexities=st.session_state.filters_complexity,
        layout_styles=st.session_state.filters_layout_style,
        primary_languages=st.session_state.filters_language,
        sparse_empty_status=actual_sparse_filter,
        segments_filter=st.session_state.filters_segments,
        defects_filter=st.session_state.filters_defects,
        show_tagging_set_only=should_filter_for_tagging_set,      # Pass the effective filter status
        tagging_set_table_exists=tagging_table_actually_exists  # Pass table existence status
    )
    st.session_state.current_page_index = 0
    st.session_state.shuffled_indices = list(range(len(st.session_state.pages_in_domain)))
    st.session_state.force_page_reset = False # Reset the flag
    if domain_changed: # If domain changed, a full rerun is good to clear everything
        st.rerun()


# Ensure pages_in_domain is populated if it's empty but domain is selected (e.g., on first load after domain selection)
# This might be redundant given the logic above but acts as a safeguard.
if st.session_state.selected_domain and not st.session_state.pages_in_domain and not (domain_changed or st.session_state.force_page_reset):
    # This condition implies filters might have resulted in zero pages, or initial load.
    # The fetch above should handle it. If still empty, the warnings below will trigger.
    pass


# --- Main Content Area ---
if not st.session_state.selected_domain:
    st.info("Please select a domain from the sidebar to view pages.")
elif not st.session_state.pages_in_domain:
    st.warning(f"No pages found for the domain: '{st.session_state.selected_domain}'.")
    st.stop()
else:
    total_pages = len(st.session_state.pages_in_domain)

    # --- Navigation Controls ---
    st.sidebar.header("Navigation")
    nav_cols = st.sidebar.columns(3)
    if nav_cols[0].button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.current_page_index == 0)):
        st.session_state.current_page_index -= 1
        st.rerun()

    if nav_cols[2].button("Next ➡️", use_container_width=True, disabled=(st.session_state.current_page_index >= total_pages - 1)):
        st.session_state.current_page_index += 1
        st.rerun()

    if nav_cols[1].button("🔀 Shuffle", use_container_width=True):
        random.shuffle(st.session_state.shuffled_indices)
        st.session_state.current_page_index = 0 # Go to the start of the new shuffled list
        st.rerun()

    st.sidebar.markdown(f"Page **{st.session_state.current_page_index + 1}** of **{total_pages}** in '{st.session_state.selected_domain}'")

    # Get current page data based on shuffled order
    actual_idx_in_original_list = st.session_state.shuffled_indices[st.session_state.current_page_index]
    current_page_info = st.session_state.pages_in_domain[actual_idx_in_original_list]
    current_page_id = current_page_info['page_identifier']
    image_s3_key = current_page_info['aws_s3_image_key']
    json_s3_key = current_page_info['aws_s3_json_key']

    st.header(f"Page ID: `{current_page_id}`")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Page Image")
        if image_s3_key:
            image_data = fetch_s3_object(s3_client, AWS_S3_BUCKET_NAME, image_s3_key, is_json=False)
            if image_data:
                st.image(image_data, use_container_width=True)
            else:
                st.warning(f"Could not load image: {image_s3_key}")
        else:
            st.warning("Image S3 key not found for this page.")

    with col2:
        st.subheader("📝 Extracted Data & Properties")
        page_details = fetch_page_details(conn, current_page_id)
        json_content = None
        if json_s3_key:
            json_content = fetch_s3_object(s3_client, AWS_S3_BUCKET_NAME, json_s3_key, is_json=True)

        if page_details:
            # Display some key properties
            st.markdown(f"**Domain:** {page_details.get('domain', 'N/A')}")
            st.markdown(f"**Source:** {page_details.get('source', 'N/A')}")
            st.markdown(f"**Orientation:** {page_details.get('orientation', 'N/A')}")
            st.markdown(f"**Color Mode:** {page_details.get('color_mode', 'N/A')}")
            st.markdown(f"**Complexity:** {page_details.get('complexity', 'N/A')}")
            st.markdown(f"**Layout Style:** {page_details.get('layout_style', 'N/A')}")
            st.markdown(f"**Primary Language:** {page_details.get('primary_language', 'N/A')}")
            st.markdown(f"**Is Sparse/Empty:** {page_details.get('is_sparse_or_empty', 'N/A')}")
            st.markdown(f"**Image S3 Key:** `{page_details.get('aws_s3_image_key', 'N/A')}`")
            st.markdown(f"**JSON S3 Key:** `{page_details.get('aws_s3_json_key', 'N/A')}`")

            with st.expander("Segments Present", expanded=False):
                if page_details.get('segments'):
                    for seg in page_details['segments']:
                        st.markdown(f"- {seg}")
                else:
                    st.markdown("_No segments recorded._")

            with st.expander("Defects Observed", expanded=False):
                if page_details.get('defects'):
                    for defect in page_details['defects']:
                        st.markdown(f"- {defect}")
                else:
                    st.markdown("_No defects recorded._")
        else:
            st.warning("Could not load page properties from database.")


        with st.expander("Full Extracted JSON", expanded=False):
            if json_content:
                st.json(json_content)
            elif json_s3_key:
                st.warning(f"Could not load JSON content from: {json_s3_key}")
            else:
                st.warning("JSON S3 key not found for this page.")

# For debugging session state
# st.sidebar.subheader("Debug Info")
# st.sidebar.json(st.session_state)
