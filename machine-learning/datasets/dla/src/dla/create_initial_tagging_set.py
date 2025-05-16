# machine-learning/datasets/dla/src/dla/create_initial_tagging_set.py
import os
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

import pandas as pd
import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

from schema import (
    DocumentDomain, DocumentSource, Complexity,
    LanguageCode, DefectType, SegmentType, PageProfile, LayoutStyle
)

# --- Configuration ---
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent.parent / '.env'
DEFAULT_SQLITE_DB_PATH = Path(__file__).resolve().parent.parent.parent / "raw_and_dedup.db"
DEFAULT_SCHEMA_SQL_PATH = Path(__file__).resolve().parent.parent.parent / "sql" / "schema.sql"

TARGET_PAGE_COUNT = 10000  # New constant for page-level target
TAGGING_SET_TABLE_NAME = "dla_selected_tagging_pages"  # New global constant

# Proposed individual domain targets (replace DOMAIN_CATEGORIES_MAP and DOMAIN_CATEGORY_TARGETS)
INDIVIDUAL_DOMAIN_TARGETS = {
    # Finance & Accounting (Total 25%)
    DocumentDomain.FINANCIAL: 0.13,
    DocumentDomain.BILLING: 0.07,
    DocumentDomain.TAX: 0.05,
    # Legal & Contracts (Total 20%)
    DocumentDomain.LEGAL: 0.12,
    DocumentDomain.GOVERNMENT: 0.08,
    # Medical & Healthcare (Total 10%)
    DocumentDomain.MEDICAL: 0.10,
    # Reports & Publications (Total 15%)
    DocumentDomain.TECHNICAL: 0.04,
    DocumentDomain.RESEARCH: 0.03,
    DocumentDomain.TEXTBOOK: 0.02,
    DocumentDomain.MAGAZINE: 0.015,
    DocumentDomain.NEWSPAPER: 0.015,
    DocumentDomain.PATENT: 0.01,
    DocumentDomain.EDUCATION: 0.02,
    # Business Operations (Total 15%)
    DocumentDomain.SUPPLY_CHAIN: 0.03,
    DocumentDomain.PROCUREMENT: 0.03,
    DocumentDomain.CONSTRUCTION: 0.03,
    DocumentDomain.REAL_ESTATE: 0.03,
    DocumentDomain.CONSULTING: 0.03,
    # Miscellaneous (Total 15%)
    DocumentDomain.HISTORICAL: 0.05,
    DocumentDomain.MISCELLANEOUS: 0.05,
    DocumentDomain.UNKNOWN: 0.05,
} # Total = 1.00

# Define Effective Complexity Categories
EFFECTIVE_COMPLEXITY_SIMPLE = "effective_simple"
EFFECTIVE_COMPLEXITY_MEDIUM = "effective_medium"
EFFECTIVE_COMPLEXITY_COMPLEX = "effective_complex"
EFFECTIVE_COMPLEXITY_VERY_COMPLEX = "effective_very_complex"

# Document characteristics targets using Effective Complexity
SUB_TARGETS_PROPORTIONS = {
    "source": {
        DocumentSource.NATIVE_DIGITAL.value: 0.40,
        DocumentSource.SCANNED_CLEAN.value: 0.20,
        DocumentSource.SCANNED_DEGRADED.value: 0.20,
        DocumentSource.PHOTO.value: 0.15,
        DocumentSource.SCREENSHOT.value: 0.05,
        DocumentSource.UNKNOWN.value: 0.0
    },
    "effective_complexity": { # Changed from "complexity"
        EFFECTIVE_COMPLEXITY_SIMPLE: 0.15,
        EFFECTIVE_COMPLEXITY_MEDIUM: 0.35,
        EFFECTIVE_COMPLEXITY_COMPLEX: 0.35,
        EFFECTIVE_COMPLEXITY_VERY_COMPLEX: 0.15,
    },
    "is_non_english": {True: 0.20, False: 0.80},
    "has_defects": {True: 0.30, False: 0.70},
}

TARGET_SEGMENT_DISTRIBUTION = {
    SegmentType.TEXT_BLOCK.value:     0.30,
    SegmentType.TABLE.value:          0.15,
    SegmentType.FORM_REGION.value:    0.12,
    SegmentType.PICTURE.value:        0.12, # Picture/Figure/Image/Chart
    SegmentType.HEADING.value:        0.05,
    SegmentType.SUBHEADING.value:     0.05,
    SegmentType.CAPTION.value:        0.05,
    SegmentType.LIST_ITEM.value:      0.05, # List Item (Bulleted/Numbered)
    SegmentType.BLOCK_QUOTE.value:    0.03, # Block Quote/Pull Quote
    SegmentType.FORMULA.value:        0.03,
    SegmentType.CODE_BLOCK.value:     0.02,
    SegmentType.LEGEND.value:         0.02,
    SegmentType.HEADER.value:         0.05,
    SegmentType.FOOTER.value:         0.05,
    SegmentType.PAGE_NUMBER.value:    0.05,
    SegmentType.FOOTNOTE.value:       0.03, # Footnote/Endnote
    SegmentType.SIGNATURE.value:      0.01, # Signature/Signature Block
    SegmentType.HANDWRITING.value:    0.01,
    SegmentType.GRAPHICAL_ITEM.value: 0.03, # Graphical Item (Logo, QR, Barcode, Stamp)
    SegmentType.UNKNOWN.value:        0.02, # Unknown/Other
}

# Hierarchies for selection preferences
SOURCE_HIERARCHY = [
    DocumentSource.PHOTO.value,
    DocumentSource.SCANNED_DEGRADED.value,
    DocumentSource.SCANNED_CLEAN.value,
    DocumentSource.NATIVE_DIGITAL.value,
    DocumentSource.SCREENSHOT.value,
    DocumentSource.UNKNOWN.value
]

# EFFECTIVE_COMPLEXITY_HIERARCHY ordered from most complex/rare to least
EFFECTIVE_COMPLEXITY_HIERARCHY = [
    EFFECTIVE_COMPLEXITY_VERY_COMPLEX,
    EFFECTIVE_COMPLEXITY_COMPLEX,
    EFFECTIVE_COMPLEXITY_MEDIUM,
    EFFECTIVE_COMPLEXITY_SIMPLE
]
# This replaces the old COMPLEXITY_HIERARCHY for selection iteration logic

# For mapping effective complexity categories to numeric scores (higher = more complex)
EFFECTIVE_COMPLEXITY_NUMERIC_MAP = {
    EFFECTIVE_COMPLEXITY_SIMPLE: 0,
    EFFECTIVE_COMPLEXITY_MEDIUM: 1,
    EFFECTIVE_COMPLEXITY_COMPLEX: 2,
    EFFECTIVE_COMPLEXITY_VERY_COMPLEX: 3
}
# Old COMPLEXITY_HIERARCHY (still needed for initial page-level complexity assessment)
# Keep the original Complexity enum and its hierarchy for determining initial max_complexity from pages
ORIGINAL_COMPLEXITY_HIERARCHY_ORDER = [
    Complexity.VERY_COMPLEX.value,
    Complexity.COMPLEX.value,
    Complexity.MEDIUM.value,
    Complexity.SIMPLE.value
]
ORIGINAL_COMPLEXITY_HIERARCHY_MAP = {
    val: idx for idx, val in enumerate(ORIGINAL_COMPLEXITY_HIERARCHY_ORDER)
}
DEFAULT_ORIGINAL_COMPLEXITY_SCORE = len(ORIGINAL_COMPLEXITY_HIERARCHY_ORDER)

# Document length configurations (will be applied to documents if needed, but primary focus is pages)
DOCUMENT_LENGTH_BINS = {
    'short': (1, 3),
    'medium': (4, 10),
    'long': (11, float('inf'))
}
# DOCUMENT_LENGTH_HIERARCHY and DOCUMENT_LENGTH_TARGETS might be less relevant for page-level selection directly,
# but could be used if we still consider parent document characteristics. For now, focus on page features.

# Clustering configuration - TO BE UPDATED FOR PAGE-LEVEL FEATURES
CLUSTER_FEATURES = [
    # 'doc_page_count', # Becomes irrelevant if we sample pages directly without doc context first
    'page_effective_complexity_numeric',
    'page_primary_source_numeric',
    'page_is_non_english_numeric',
    'page_has_defects_numeric',
    'page_distinct_defect_types_count',
    'page_distinct_segment_types_count',
    'page_has_table_segment_numeric',
    'page_has_form_segment_numeric',
    'page_has_picture_segment_numeric'
    # Add other relevant page-level numeric features from PageProfile if needed
]

# The new calculate_page_effective_complexity function should be here if not already moved
# Ensure calculate_page_effective_complexity is defined before main or wherever it's first called.

# Ensure DOCUMENT_LENGTH_HIERARCHY, N_CLUSTERS_PER_DOMAIN are defined if select_pages_for_tagging will need them
# For now, they are not used by the current main() logic up to the temporary sampling.
DOCUMENT_LENGTH_HIERARCHY = ['long', 'medium', 'short'] # Keep if page-level logic might use doc length context

# Define N_CLUSTERS_PER_PAGE_DOMAIN_CATEGORY if not already defined or if it was N_CLUSTERS_PER_DOMAIN
N_CLUSTERS_PER_PAGE_DOMAIN_CATEGORY = 15

class DatabaseConnectionError(Exception):
    """Custom exception for database connection issues."""
    pass

def load_environment() -> None:
    """Load environment variables from .env file."""
    if DEFAULT_ENV_PATH.is_file():
        load_dotenv(DEFAULT_ENV_PATH, override=True)
    else:
        print(f"Warning: .env file not found at {DEFAULT_ENV_PATH}")

def connect_sqlite(db_path: Path) -> sqlite3.Connection:
    """Connect to SQLite database."""
    try:
        return sqlite3.connect(db_path)
    except sqlite3.Error as e:
        raise DatabaseConnectionError(f"Failed to connect to SQLite database: {e}")

def connect_postgres(db_url: str) -> psycopg.Connection:
    """Connect to PostgreSQL database."""
    try:
        return psycopg.connect(db_url)
    except psycopg.Error as e:
        raise DatabaseConnectionError(f"Failed to connect to PostgreSQL database: {e}")

def get_length_bin(page_count: int) -> str:
    """Determine document length bin based on page count."""
    for bin_name, (min_pages, max_pages) in DOCUMENT_LENGTH_BINS.items():
        if min_pages <= page_count <= max_pages:
            return bin_name
    return 'short'  # Default to short if no bin matches

def calculate_page_effective_complexity(page_profile: Dict[str, Any]) -> Tuple[str, int]:
    """
    Calculates the effective complexity category and numeric score for a single page.
    """
    original_complexity_val = page_profile.get('complexity', Complexity.SIMPLE.value)
    layout_style = page_profile.get('layout_style') # Can be None

    effective_category = EFFECTIVE_COMPLEXITY_SIMPLE # Default

    # Rule-based assignment for effective complexity based on page's own layout and original complexity
    if layout_style in [LayoutStyle.FORM_LIKE.value, LayoutStyle.COMPLEX.value]:
        if original_complexity_val == Complexity.VERY_COMPLEX.value:
            effective_category = EFFECTIVE_COMPLEXITY_VERY_COMPLEX
        elif original_complexity_val == Complexity.COMPLEX.value:
            effective_category = EFFECTIVE_COMPLEXITY_VERY_COMPLEX
        else: # Simple or Medium original complexity
            effective_category = EFFECTIVE_COMPLEXITY_COMPLEX
    elif layout_style == LayoutStyle.MULTI_COLUMN.value:
        if original_complexity_val == Complexity.VERY_COMPLEX.value:
            effective_category = EFFECTIVE_COMPLEXITY_VERY_COMPLEX
        elif original_complexity_val == Complexity.COMPLEX.value:
            effective_category = EFFECTIVE_COMPLEXITY_COMPLEX
        elif original_complexity_val == Complexity.MEDIUM.value:
            effective_category = EFFECTIVE_COMPLEXITY_COMPLEX
        else: # Simple
            effective_category = EFFECTIVE_COMPLEXITY_MEDIUM
    elif layout_style in [LayoutStyle.DOUBLE_COLUMN.value, LayoutStyle.IMAGE_HEAVY.value]:
        if original_complexity_val == Complexity.VERY_COMPLEX.value:
            effective_category = EFFECTIVE_COMPLEXITY_VERY_COMPLEX
        elif original_complexity_val == Complexity.COMPLEX.value:
            effective_category = EFFECTIVE_COMPLEXITY_COMPLEX
        elif original_complexity_val == Complexity.MEDIUM.value:
            effective_category = EFFECTIVE_COMPLEXITY_MEDIUM
        else: # Simple
            if layout_style == LayoutStyle.IMAGE_HEAVY.value:
                effective_category = EFFECTIVE_COMPLEXITY_MEDIUM
            else: # Double column simple
                effective_category = EFFECTIVE_COMPLEXITY_SIMPLE
    else: # Default to original complexity mapping if no strong layout cues or layout_style is None/Unknown
        if original_complexity_val == Complexity.VERY_COMPLEX.value:
            effective_category = EFFECTIVE_COMPLEXITY_VERY_COMPLEX
        elif original_complexity_val == Complexity.COMPLEX.value:
            effective_category = EFFECTIVE_COMPLEXITY_COMPLEX
        elif original_complexity_val == Complexity.MEDIUM.value:
            effective_category = EFFECTIVE_COMPLEXITY_MEDIUM
        else: # Simple
            effective_category = EFFECTIVE_COMPLEXITY_SIMPLE

    numeric_score = EFFECTIVE_COMPLEXITY_NUMERIC_MAP.get(effective_category, 0)
    return effective_category, numeric_score

def fetch_doc_page_mapping_sqlite(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fetch document-page mapping from SQLite database (raw_and_dedup.db).
    This function now assumes 'raw_and_dedup.db' contains a 'pages_raw' table
    with 'doc_id' and 'dedup_id' (which will be used as 'page_identifier').
    """
    query = """
    SELECT 
        doc_id,
        dedup_id AS page_identifier  -- Use dedup_id from pages_raw as the page_identifier
    FROM pages_raw
    WHERE dedup_id IS NOT NULL; -- Ensure we only get pages that have a deduplication ID
    """
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            print("Warning: No document-page mapping found in SQLite (pages_raw table might be empty or all dedup_ids are NULL).")
        # Basic validation
        if not {'doc_id', 'page_identifier'}.issubset(df.columns):
            raise ValueError("Query result from pages_raw is missing 'doc_id' or 'page_identifier' columns.")
        print(f"Fetched {len(df)} page mappings from pages_raw in SQLite.")
        return df
    except pd.errors.DatabaseError as e:
        if "no such table: pages_raw" in str(e):
            print("ERROR: The table 'pages_raw' does not exist in the SQLite database.")
            print("Please ensure 'src/dla/main.py' has been run successfully to create and populate 'raw_and_dedup.db'.")
        else:
            print(f"ERROR: Database error while querying 'pages_raw' in SQLite: {e}")
        raise # Re-raise the exception to halt execution

def fetch_page_profiles_postgres(conn: psycopg.Connection, page_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch page profiles from PostgreSQL database."""
    profiles = {}
    with conn.cursor(row_factory=dict_row) as cur:
        # Fetch in chunks to avoid memory issues
        chunk_size = 1000 
        for i in range(0, len(page_ids), chunk_size):
            chunk = page_ids[i:i + chunk_size]
            cur.execute(
                "SELECT * FROM page_profiles WHERE page_identifier = ANY(%s);",
                (chunk,)
            )
            for row in cur.fetchall():
                profiles[row['page_identifier']] = dict(row)
    return profiles

# def aggregate_document_profiles(...): # Entire function body commented out or removed
#     pass # Placeholder if needed, or just remove completely if no longer called.

# def select_documents_for_tagging(...): # Entire function body commented out or removed
#     pass # Placeholder if needed, or just remove completely if no longer called.

def store_selected_pages_in_source_db(
    pg_conn: psycopg.Connection,
    selected_page_identifiers: List[str]
) -> None:
    """
    Stores the selected page identifiers in a dedicated table within the source PostgreSQL database.
    This table will be used by other tools (e.g., DLA Explorer) to filter for the tagging set.
    """
    if not selected_page_identifiers:
        print("No page identifiers provided to store. Skipping table creation/update.")
        return

    try:
        with pg_conn.cursor() as cur:
            # Create the table if it doesn't exist
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    page_identifier TEXT PRIMARY KEY
                );
            """).format(sql.Identifier(TAGGING_SET_TABLE_NAME)))
            print(f"Ensured table '{TAGGING_SET_TABLE_NAME}' exists in the source database.")

            # Clear existing entries in the table to refresh the set
            cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(TAGGING_SET_TABLE_NAME)))
            print(f"Cleared existing entries from '{TAGGING_SET_TABLE_NAME}'.")

            # Insert new selected page identifiers
            insert_query = sql.SQL("INSERT INTO {} (page_identifier) VALUES (%s) ON CONFLICT (page_identifier) DO NOTHING;").format(
                sql.Identifier(TAGGING_SET_TABLE_NAME)
            )
            data_to_insert = [(pid,) for pid in selected_page_identifiers]
            
            cur.executemany(insert_query, data_to_insert)
            pg_conn.commit() 
            print(f"Successfully inserted/updated {len(selected_page_identifiers)} page identifiers into '{TAGGING_SET_TABLE_NAME}'.")

    except psycopg.Error as e:
        if pg_conn and not pg_conn.closed:
            try:
                pg_conn.rollback() 
            except Exception as rb_e:
                print(f"Error during rollback: {rb_e}")
        print(f"Error interacting with table '{TAGGING_SET_TABLE_NAME}': {e}")
        raise 

# --- New Function: assign_segment_priority_scores ---
def assign_segment_priority_scores(
    pages_df: pd.DataFrame,
    target_dist: Dict[str, float]
) -> pd.DataFrame:
    """
    Assigns a 'segment_priority_score' to each page based on how much it contributes
    to achieving the target segment distribution.
    Pages with segments that are underrepresented in the overall dataset relative to targets
    will receive higher scores.
    This function MODIFIES pages_df by adding a new column.
    """
    print("\n--- Assigning Segment Priority Scores ---")
    if pages_df.empty:
        print("Pages DataFrame is empty. Skipping segment priority scoring.")
        pages_df['segment_priority_score'] = 0 # Ensure column exists even if empty
        return pages_df

    if 'page_segments' not in pages_df.columns:
        print("Warning: 'page_segments' column missing. Cannot assign segment priority scores.")
        pages_df['segment_priority_score'] = 0 # Default score
        return pages_df

    # Calculate initial proportions of each segment in the entire dataset
    all_segments_list = []    
    for seg_list in pages_df['page_segments'].dropna(): # Ensure seg_list is iterable
        if isinstance(seg_list, list):
            all_segments_list.extend(seg_list)
    
    initial_segment_counts = defaultdict(int)
    for seg in all_segments_list:
        initial_segment_counts[seg] += 1
    
    total_segments_in_dataset = len(all_segments_list)
    if total_segments_in_dataset == 0:
        print("No segments found in dataset. Assigning zero priority score to all pages.")
        pages_df['segment_priority_score'] = 0.0
        return pages_df

    initial_segment_proportions = {
        seg: count / total_segments_in_dataset
        for seg, count in initial_segment_counts.items()
    }
    print(f"Initial segment proportions in the full dataset: {initial_segment_proportions}")
    print(f"Target segment distribution: {target_dist}")

    # Calculate scores
    priority_scores = []
    for index, row in pages_df.iterrows():
        page_score = 0.0
        page_segments = row.get('page_segments')
        if isinstance(page_segments, list):
            for segment_type in page_segments:
                target_proportion = target_dist.get(segment_type, 0)
                current_proportion = initial_segment_proportions.get(segment_type, 0)
                need_for_segment = target_proportion - current_proportion
                
                # Reward pages that have segments the dataset needs more of to reach target distribution
                if need_for_segment > 0:
                    # Simple bonus: add 1 for each needed segment type.
                    # Could be weighted by how much it's needed (need_for_segment)
                    # or by the target_proportion itself to prioritize generally important segments.
                    page_score += 1.0 # Basic score: presence of a needed segment
                    # Example of weighted score: page_score += need_for_segment 
                    # Example: page_score += target_proportion (favors segments with high targets)

        priority_scores.append(page_score)
    
    pages_df['segment_priority_score'] = priority_scores
    print(f"Assigned 'segment_priority_score'. Min: {pages_df['segment_priority_score'].min()}, Max: {pages_df['segment_priority_score'].max()}, Mean: {pages_df['segment_priority_score'].mean():.2f}")
    print("---------------------------------------\n")
    return pages_df

def select_pages_for_tagging(
    pages_df: pd.DataFrame,
    target_total_page_count: int,
    domain_targets: Dict[DocumentDomain, float],
    random_state: int = 42,
    complexity_weight: float = 1.0, # Weight for complexity score
    segment_priority_weight: float = 1.5 # Weight for segment priority score
) -> List[str]:
    """
    Selects pages for tagging based on stratified sampling according to domain targets,
    using an 80/20 strategy within each domain. The 80% are prioritized based on a 
    combined score of effective complexity and segment priority.

    Args:
        pages_df: DataFrame containing page information, including 'page_identifier',
                  'page_domain', 'page_effective_complexity_numeric',
                  and 'segment_priority_score'.
        target_total_page_count: The total number of pages desired for the tagging set.
        domain_targets: A dictionary mapping DocumentDomain enums or their string values
                        to target proportions.
        random_state: Seed for random sampling.
        complexity_weight: Weight for 'page_effective_complexity_numeric' in combined score.
        segment_priority_weight: Weight for 'segment_priority_score' in combined score.

    Returns:
        A list of selected page identifiers.
    """
    if pages_df.empty or target_total_page_count == 0:
        print("Input DataFrame is empty or target count is zero. Returning empty list.")
        return []

    required_cols = ['page_identifier', 'page_domain', 
                     'page_effective_complexity_numeric', 'segment_priority_score']
    if not all(col in pages_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in pages_df.columns]
        raise ValueError(f"Missing required columns in pages_df for selection: {missing_cols}")

    # Make a copy to avoid modifying the original DataFrame passed to the function, especially for new columns
    processing_df = pages_df.copy()

    # Calculate combined priority score for the 80% selection stage
    # Ensure scores are numeric and handle potential NaNs by filling with a low value (e.g., 0)
    processing_df['page_effective_complexity_numeric'] = pd.to_numeric(processing_df['page_effective_complexity_numeric'], errors='coerce').fillna(0)
    processing_df['segment_priority_score'] = pd.to_numeric(processing_df['segment_priority_score'], errors='coerce').fillna(0)
    
    processing_df['combined_priority_score'] = (
        processing_df['page_effective_complexity_numeric'] * complexity_weight +
        processing_df['segment_priority_score'] * segment_priority_weight
    )
    print(f"Calculated 'combined_priority_score'. Min: {processing_df['combined_priority_score'].min():.2f}, Max: {processing_df['combined_priority_score'].max():.2f}, Mean: {processing_df['combined_priority_score'].mean():.2f}")

    selected_page_ids_set = set()
    final_selected_pages_list = []
    # Use processing_df which contains the combined_priority_score for selections
    available_pages_df = processing_df 

    print("\n--- Starting Domain-Level Stratified Selection (80/20 Strategy with Combined Score) ---")
    np.random.seed(random_state)
    random.seed(random_state)

    domain_page_targets = {}
    for domain, proportion in domain_targets.items():
        domain_value = domain.value if isinstance(domain, DocumentDomain) else domain
        if proportion > 0:
             domain_page_targets[domain_value] = round(target_total_page_count * proportion)

    current_total_target = sum(domain_page_targets.values())
    diff = target_total_page_count - current_total_target
    if diff != 0:
        adjust_domains = [d for d, t in domain_page_targets.items() if t > 0]
        if adjust_domains:
            adjust_domains.sort()
            for i in range(abs(diff)):
                domain_to_adjust = adjust_domains[i % len(adjust_domains)]
                domain_page_targets[domain_to_adjust] += 1 if diff > 0 else -1
                domain_page_targets[domain_to_adjust] = max(0, domain_page_targets[domain_to_adjust])

    print(f"Calculated target counts per domain: {domain_page_targets}")
    print(f"Adjusted total target across domains: {sum(domain_page_targets.values())}")

    sorted_domains = sorted(domain_page_targets.keys())

    for domain_value in sorted_domains:
        n_target = domain_page_targets[domain_value]
        if n_target <= 0:
            print(f"Skipping domain '{domain_value}' as target count is {n_target}.")
            continue

        domain_pages = available_pages_df[available_pages_df['page_domain'] == domain_value]
        num_available_in_domain = len(domain_pages)
        print(f"Domain '{domain_value}': Target={n_target}, Available={num_available_in_domain}")

        if num_available_in_domain == 0:
            print(f"  Warning: No available pages for domain '{domain_value}'. Cannot meet target.")
            continue

        num_to_sample_total = min(n_target, num_available_in_domain)
        n_target_first_stage = round(num_to_sample_total * 0.8)
        # n_target_second_stage = num_to_sample_total - n_target_first_stage # Not strictly needed to store

        print(f"  Calculated split: {n_target_first_stage} prioritized (target), {num_to_sample_total - n_target_first_stage} random (target).")

        # --- Stage 1: Prioritized Selection (Top 80% target) using combined_priority_score ---
        sorted_domain_pages = domain_pages.sort_values(
            by=['combined_priority_score', 'page_identifier'], # Sort by new combined score
            ascending=[False, True]  # Higher score first, then consistent ID order
        )

        first_stage_selection_df = sorted_domain_pages.head(n_target_first_stage)
        first_stage_ids = first_stage_selection_df['page_identifier'].tolist()
        num_selected_first_stage = len(first_stage_ids)
        print(f"  Stage 1 (Prioritized): Selected {num_selected_first_stage} pages using combined score (target was {n_target_first_stage}).")

        remaining_domain_pages_df = domain_pages.drop(first_stage_selection_df.index)
        num_remaining_in_domain = len(remaining_domain_pages_df)
        num_needed_randomly = num_to_sample_total - num_selected_first_stage
        print(f"  Stage 2 (Random): Need to select {num_needed_randomly} more pages randomly from the remaining {num_remaining_in_domain} pages in this domain.")

        num_to_sample_randomly = min(num_needed_randomly, num_remaining_in_domain)
        second_stage_selection_df = pd.DataFrame()
        if num_to_sample_randomly > 0:
            second_stage_selection_df = remaining_domain_pages_df.sample(
                n=num_to_sample_randomly, random_state=random_state, replace=False
            )
            second_stage_ids = second_stage_selection_df['page_identifier'].tolist()
            print(f"  Stage 2 (Random): Selected {len(second_stage_ids)} pages.")
        else:
            second_stage_ids = []
            print(f"  Stage 2 (Random): No pages needed or available for random selection.")

        combined_domain_ids = first_stage_ids + second_stage_ids
        total_selected_for_domain = len(combined_domain_ids)
        print(f"  Total selected for domain '{domain_value}': {total_selected_for_domain} pages.")

        newly_selected_count = 0
        for page_id in combined_domain_ids:
             if page_id not in selected_page_ids_set:
                 selected_page_ids_set.add(page_id)
                 final_selected_pages_list.append(page_id)
                 newly_selected_count += 1
        
        # Critical: Update available_pages_df by removing ALL selected pages for this domain
        # from the main pool used for inter-domain iteration and final shortfall filling.
        # Indices must come from the `available_pages_df`'s original index if it was a slice, 
        # but since `domain_pages` is a slice of `available_pages_df`, their indices are consistent.
        indices_to_drop_from_available = first_stage_selection_df.index.union(second_stage_selection_df.index)
        available_pages_df = available_pages_df.drop(indices_to_drop_from_available)

        if newly_selected_count != total_selected_for_domain:
             print(f"  Warning: Some pages selected for domain '{domain_value}' might have been duplicates if logic changes; currently should be unique per domain pass.")

    current_selected_count = len(selected_page_ids_set)
    overall_shortfall = target_total_page_count - current_selected_count
    print(f"\nTotal pages selected after domain stratification: {current_selected_count}")
    print(f"Overall target: {target_total_page_count}, Shortfall: {overall_shortfall}")

    if overall_shortfall > 0:
        print(f"Attempting to fill overall shortfall of {overall_shortfall} pages randomly from global remaining pool...")
        num_remaining_globally = len(available_pages_df)
        print(f"  Available pages in the global remaining pool: {num_remaining_globally}")
        num_to_fill_globally = min(overall_shortfall, num_remaining_globally)

        if num_to_fill_globally > 0:
            # available_pages_df now correctly represents pages not picked by any domain specific logic
            filler_pages = available_pages_df.sample(n=num_to_fill_globally, random_state=random_state, replace=False)
            filler_ids = filler_pages['page_identifier'].tolist()
            newly_filled_count = 0
            for page_id in filler_ids:
                 if page_id not in selected_page_ids_set: # Should always be true if available_pages_df is managed correctly
                     selected_page_ids_set.add(page_id)
                     final_selected_pages_list.append(page_id)
                     newly_filled_count += 1
            print(f"  Selected {newly_filled_count} additional pages randomly from global pool to fill shortfall.")
            available_pages_df = available_pages_df.drop(filler_pages.index)
        else:
            print("  No remaining pages available globally to fill the shortfall, or shortfall is already zero.")

    final_selected_count = len(selected_page_ids_set)
    if len(final_selected_pages_list) != final_selected_count:
        print(f"Warning: Mismatch between list length ({len(final_selected_pages_list)}) and set size ({final_selected_count}). Recreating list from set.")
        final_selected_pages_list = list(selected_page_ids_set)

    if final_selected_count > target_total_page_count:
        print(f"Warning: Selected {final_selected_count} pages, which is more than target {target_total_page_count}. Trimming randomly...")
        random.shuffle(final_selected_pages_list)
        final_selected_pages_list = final_selected_pages_list[:target_total_page_count]
        print(f"  Trimmed selection down to {len(final_selected_pages_list)} pages.")
    elif final_selected_count < target_total_page_count:
         print(f"Warning: Final selected count ({final_selected_count}) is less than target ({target_total_page_count}). Could not find enough pages.")
    else:
        random.shuffle(final_selected_pages_list)

    print(f"\n--- Domain-Level Stratified Selection Complete ---")
    print(f"Final selected page count: {len(final_selected_pages_list)}")

    if len(set(final_selected_pages_list)) != len(final_selected_pages_list):
         print("ERROR: Duplicates detected in the final list after all steps. This should not happen.")
         final_selected_pages_list = list(set(final_selected_pages_list))

    return final_selected_pages_list

def main() -> None:
    """Main function to orchestrate the dataset creation."""
    sqlite_conn = None
    source_pg_conn = None

    try:
        load_environment()
        source_db_url = os.getenv("SOURCE_DATABASE_URL")
        sqlite_db_path_str = os.getenv("SQLITE_DB_PATH", str(DEFAULT_SQLITE_DB_PATH))
        sqlite_db_path = Path(sqlite_db_path_str)

        if not source_db_url:
            raise ValueError("Missing SOURCE_DATABASE_URL environment variable.")

        sqlite_conn = connect_sqlite(sqlite_db_path)
        source_pg_conn = connect_postgres(str(source_db_url))
        
        try:
            doc_page_map_df = fetch_doc_page_mapping_sqlite(sqlite_conn)
            if doc_page_map_df.empty:
                print("No document-page mapping found in SQLite. Exiting.")
                return

            all_page_identifiers_from_sqlite = doc_page_map_df['page_identifier'].unique().tolist()
            page_profiles_dict = fetch_page_profiles_postgres(
                source_pg_conn,
                all_page_identifiers_from_sqlite
            )
            if not page_profiles_dict:
                print("No page profiles fetched from PostgreSQL. Exiting.")
                return

            # --- Build page-level DataFrame (pages_df) ---
            print("Building page-level DataFrame (pages_df)...")
            page_data_list = []
            
            # Create a map of page_identifier to doc_id for quick lookup
            page_to_doc_id_map = pd.Series(doc_page_map_df.doc_id.values, index=doc_page_map_df.page_identifier).to_dict()

            source_hierarchy_map = {val: idx for idx, val in enumerate(SOURCE_HIERARCHY)}

            for page_id, profile in page_profiles_dict.items():
                if profile is None: 
                    print(f"Warning: Skipping page_id {page_id} due to None profile.")
                    continue

                page_info: Dict[str, Any] = {'page_identifier': page_id}
                page_info['doc_id'] = page_to_doc_id_map.get(page_id, "UNKNOWN_DOC_ID")

                # Directly add attributes from page_profile
                page_info['page_domain'] = profile.get('domain', DocumentDomain.UNKNOWN.value)
                page_info['page_source'] = profile.get('source', DocumentSource.UNKNOWN.value)
                page_info['page_primary_source_numeric'] = source_hierarchy_map.get(page_info['page_source'], len(SOURCE_HIERARCHY))
                
                page_info['page_original_complexity'] = profile.get('complexity', Complexity.SIMPLE.value)
                page_info['page_layout_style'] = profile.get('layout_style', LayoutStyle.UNKNOWN.value)

                # Calculate page-level effective complexity
                eff_cat, eff_num = calculate_page_effective_complexity(profile)
                page_info['page_effective_complexity_category'] = eff_cat
                page_info['page_effective_complexity_numeric'] = eff_num

                page_info['page_is_non_english'] = profile.get('primary_language') not in [LanguageCode.ENGLISH.value, LanguageCode.UNKNOWN.value, None]
                page_info['page_is_non_english_numeric'] = int(page_info['page_is_non_english'])
                page_info['page_primary_language'] = profile.get('primary_language', LanguageCode.UNKNOWN.value)

                # Page-level Segments
                page_segments_raw = profile.get('segments', [])
                page_segments_values = set()
                if isinstance(page_segments_raw, list):
                     page_segments_values = {str(s.value) if hasattr(s, 'value') else str(s) for s in page_segments_raw}
                page_info['page_segments'] = list(page_segments_values)
                
                page_info['page_distinct_segment_types_count'] = len(page_segments_values)
                page_info['page_has_table_segment'] = SegmentType.TABLE.value in page_segments_values
                page_info['page_has_table_segment_numeric'] = int(page_info['page_has_table_segment'])
                page_info['page_has_form_segment'] = SegmentType.FORM_REGION.value in page_segments_values
                page_info['page_has_form_segment_numeric'] = int(page_info['page_has_form_segment'])
                page_info['page_has_picture_segment'] = SegmentType.PICTURE.value in page_segments_values
                page_info['page_has_picture_segment_numeric'] = int(page_info['page_has_picture_segment'])

                # Page-level Defects
                page_defects_raw = profile.get('defects', [])
                page_defects_values = set()
                if isinstance(page_defects_raw, list):
                     page_defects_values = {str(d.value) if hasattr(d, 'value') else str(d) for d in page_defects_raw}

                page_info['page_has_defects'] = bool(page_defects_values)
                page_info['page_has_defects_numeric'] = int(page_info['page_has_defects'])
                page_info['page_distinct_defect_types_count'] = len(page_defects_values)
                
                # Add other scalar attributes from profile directly if needed for CLUSTER_FEATURES
                # e.g., page_info['is_sparse_or_empty'] = profile.get('is_sparse_or_empty', False)

                page_data_list.append(page_info)

            pages_df = pd.DataFrame(page_data_list)

            if pages_df.empty:
                print("No page data was processed into pages_df. Exiting.")
                return

            print(f"Successfully built pages_df with {len(pages_df)} pages.")
            print(f"Columns in pages_df: {pages_df.columns.tolist()}") # Verify columns

            print("\n--- Initial Distribution of Page Domain Categories in pages_df (before any selection) ---")
            if 'page_domain' in pages_df.columns:
                print("Raw counts:")
                initial_domain_counts = pages_df['page_domain'].value_counts(normalize=False)
                print(initial_domain_counts)
                # Check for HISTORICAL specifically
                historical_domain_value = DocumentDomain.HISTORICAL.value # Assuming 'page_domain' stores string values of enums
                if historical_domain_value in initial_domain_counts:
                    print(f"Initial count for HISTORICAL in pages_df: {initial_domain_counts[historical_domain_value]}")
                else:
                    print(f"HISTORICAL domain ('{historical_domain_value}') not found in initial pages_df domain counts.")

                print("\nNormalized (proportions):")
                print(pages_df['page_domain'].value_counts(normalize=True))
            else:
                print("'page_domain' column not found in pages_df.")
            print("--------------------------------------------------------------------\n")

            # --- Assign Segment Priority Scores (New Step 1) ---
            # This function now adds a score column to pages_df and returns the modified df.
            pages_df_with_priority = assign_segment_priority_scores(
                pages_df.copy(), # Operate on a copy to be safe, or ensure pages_df is not used elsewhere unmodified
                TARGET_SEGMENT_DISTRIBUTION
            )

            # The pre_balance_by_segment_type function is now replaced by assign_segment_priority_scores.
            # We no longer use its output as the primary input for domain selection if it filtered rows.
            # The select_pages_for_tagging function will now use pages_df_with_priority.

            # Diagnostic print: what are the counts in the df being passed to selection?
            print("\n--- Distribution of Page Domain Categories in pages_df_with_priority (after priority scoring, before domain selection) ---")
            if 'page_domain' in pages_df_with_priority.columns:
                priority_domain_counts = pages_df_with_priority['page_domain'].value_counts(normalize=False)
                print("Raw counts:")
                print(priority_domain_counts)
                historical_domain_value = DocumentDomain.HISTORICAL.value
                if historical_domain_value in priority_domain_counts:
                    print(f"Count for HISTORICAL in pages_df_with_priority: {priority_domain_counts[historical_domain_value]}")
                else:
                    print(f"HISTORICAL domain ('{historical_domain_value}') not found in pages_df_with_priority.")
                print("\nNormalized (proportions):")
                print(pages_df_with_priority['page_domain'].value_counts(normalize=True))
            else:
                print("'page_domain' column not found in pages_df_with_priority.")
            print("----------------------------------------------------------------------------\n")
            
            selection_input_df = pages_df_with_priority # This is the DataFrame to be used for selection

            # --- Call the page-level selection function (Step 2) ---
            # select_pages_for_tagging will be modified later to USE the 'segment_priority_score'
            print(f"\n--- Starting final page selection process on DataFrame with {len(selection_input_df)} pages ---")
            final_selected_page_identifiers = select_pages_for_tagging(
                selection_input_df, # This now contains segment_priority_score
                TARGET_PAGE_COUNT,
                INDIVIDUAL_DOMAIN_TARGETS,
                random_state=42
            )
            
            if not final_selected_page_identifiers:
                print("No pages were selected. Exiting.")
                return

            print(f"Final selected page count for tagging set: {len(final_selected_page_identifiers)}")

            # Summary based on the selected pages (can be enhanced later)
            final_selected_pages_df = pages_df[pages_df['page_identifier'].isin(final_selected_page_identifiers)]
            print("\n--- Final Selected Pages Summary ---")
            if not final_selected_pages_df.empty:
                print("\nDomain Category Distribution (of selected pages):")
                print(final_selected_pages_df['page_domain'].value_counts(normalize=True))
                print("\nEffective Complexity Distribution (of selected pages):")
                print(final_selected_pages_df['page_effective_complexity_category'].value_counts(normalize=True))
                # Add more summary stats as needed
            else:
                print("No pages in the final selection for summary.")

            store_selected_pages_in_source_db(source_pg_conn, final_selected_page_identifiers)
            print(f"Selected page identifiers stored in table '{TAGGING_SET_TABLE_NAME}' in the source database.")

        except Exception as inner_e:
            print(f"Error during data processing, selection, or storage: {inner_e}")
            raise 

    except DatabaseConnectionError as db_conn_e:
        print(f"Database connection error: {db_conn_e}")
    except ValueError as val_e:
        print(f"Configuration or value error: {val_e}")
    except Exception as e: 
        print(f"An unexpected error occurred in main execution: {e}")
        raise 

    finally:
        print("Closing database connections...")
        if sqlite_conn:
            try:
                sqlite_conn.close()
                print("SQLite connection closed.")
            except Exception as e_sqlite_close:
                print(f"Error closing SQLite connection: {e_sqlite_close}")
        
        if source_pg_conn and not source_pg_conn.closed:
            try:
                source_pg_conn.close()
                print("Source PostgreSQL connection closed.")
            except Exception as e_pg_source_close:
                print(f"Error closing source PostgreSQL connection: {e_pg_source_close}")

        print("Database cleanup finished.")

if __name__ == "__main__":
    main()