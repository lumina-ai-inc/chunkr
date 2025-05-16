-- =============================================
-- Main table to store page profile attributes
-- =============================================
CREATE TABLE IF NOT EXISTS page_profiles (
    page_identifier TEXT PRIMARY KEY, -- Unique identifier (dedup_id) for the page
    domain TEXT NOT NULL CHECK (domain IN (
        'financial', 'billing', 'tax', 'supply_chain', 'technical', 'research',
        'legal', 'government', 'procurement', 'consulting', 'magazine', 'newspaper',
        'textbook', 'historical', 'patent', 'education', 'medical', 'real_estate',
        'construction', 'miscellaneous', 'unknown'
    )),
    source TEXT NOT NULL CHECK (source IN (
        'native_digital', 'scanned_clean', 'scanned_degraded', 'photo', 'screenshot', 'unknown'
    )),
    orientation TEXT NOT NULL CHECK (orientation IN (
        'portrait', 'landscape', 'mixed', 'unknown'
    )),
    color_mode TEXT NOT NULL CHECK (color_mode IN (
        'color', 'grayscale', 'black_and_white', 'unknown'
    )),
    complexity TEXT NOT NULL CHECK (complexity IN (
        'Simple', 'Medium', 'Complex', 'Very Complex'
    )),
    layout_style TEXT NOT NULL CHECK (layout_style IN (
        'Single Column', 'Double Column', 'Multiple Columns', 'Complex/Mixed',
        'Image Heavy', 'Form-like', 'Unknown'
    )),
    primary_language TEXT NOT NULL CHECK (primary_language IN (
        'en', 'es', 'fr', 'de', 'nl', 'it', 'pt', 'pl', 'sv', 'tr', 'el', 'ru',
        'zh', 'ja', 'ko', 'ar', 'he', 'hi', 'bn', 'ur', 'ta', 'fa', 'vi', 'th',
        'id', 'ms', 'tl', 'mixed', 'other', 'unknown'
    )),
    is_sparse_or_empty BOOLEAN NOT NULL,
    aws_s3_json_key TEXT NOT NULL UNIQUE, -- S3 key for the source JSON data
    aws_s3_image_key TEXT NOT NULL UNIQUE, -- S3 key for the source image (assuming you can derive this)
    extraction_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP -- Timestamp of insertion/update
);

-- Add comments to the columns for better understanding
COMMENT ON COLUMN page_profiles.page_identifier IS 'Unique identifier (dedup_id) for the page, derived during processing.';
COMMENT ON COLUMN page_profiles.domain IS 'Primary subject matter classification of the document page.';
COMMENT ON COLUMN page_profiles.source IS 'Origin or method of creation/capture of the document page.';
COMMENT ON COLUMN page_profiles.orientation IS 'Intended orientation of the page content.';
COMMENT ON COLUMN page_profiles.color_mode IS 'Color representation of the document page image.';
COMMENT ON COLUMN page_profiles.complexity IS 'Estimated layout/content complexity.';
COMMENT ON COLUMN page_profiles.layout_style IS 'Overall page layout style.';
COMMENT ON COLUMN page_profiles.primary_language IS 'Primary language identified on the page (ISO 639-1 code, mixed, other, unknown).';
COMMENT ON COLUMN page_profiles.is_sparse_or_empty IS 'True if the page has very little meaningful content.';
COMMENT ON COLUMN page_profiles.aws_s3_json_key IS 'Full S3 key for the extracted JSON metadata file in the AWS bucket.';
COMMENT ON COLUMN page_profiles.aws_s3_image_key IS 'Full S3 key for the source page image file (likely in GCS, adjust if needed).'; -- Adjust comment/column if image source differs
COMMENT ON COLUMN page_profiles.extraction_timestamp IS 'Timestamp when this record was created or last updated in the database.';

-- Optional: Add indexes for frequently queried columns if needed (e.g., domain, source)
CREATE INDEX IF NOT EXISTS idx_page_profiles_domain ON page_profiles (domain);
CREATE INDEX IF NOT EXISTS idx_page_profiles_source ON page_profiles (source);
CREATE INDEX IF NOT EXISTS idx_page_profiles_language ON page_profiles (primary_language);


-- =============================================
-- Junction table for page segments (Many-to-Many)
-- =============================================
CREATE TABLE IF NOT EXISTS page_segments (
    page_identifier TEXT NOT NULL,
    segment_type TEXT NOT NULL CHECK (segment_type IN (
        'Title', 'Heading', 'Subheading', 'Code Block', 'List Item (Bulleted/Numbered)',
        'Block Quote/Pull Quote', 'Caption', 'Legend', 'Formula', 'Header', 'Footer',
        'Page Number', 'Footnote/Endnote', 'Picture/Figure/Image/Chart', 'Table',
        'Text Block/Paragraph', 'Form Region/Key-Value Pair', 'Signature/Signature Block',
        'Handwriting', 'Graphical Item (Logo, QR, Barcode, Stamp)', 'Unknown/Other'
    )),
    -- Composite primary key ensures each segment type is listed only once per page
    PRIMARY KEY (page_identifier, segment_type),
    -- Foreign key constraint ensures page_identifier exists in the main table
    -- ON DELETE CASCADE means if a page_profile is deleted, its corresponding segments are also deleted
    CONSTRAINT fk_page_identifier
        FOREIGN KEY(page_identifier)
        REFERENCES page_profiles(page_identifier)
        ON DELETE CASCADE
);

-- Add comments
COMMENT ON TABLE page_segments IS 'Junction table mapping pages to the types of content segments present on them.';
COMMENT ON COLUMN page_segments.page_identifier IS 'Foreign key referencing the unique page identifier in page_profiles.';
COMMENT ON COLUMN page_segments.segment_type IS 'The type of content segment observed on the page.';

-- Index for faster lookups based on segment type (optional, but potentially useful)
CREATE INDEX IF NOT EXISTS idx_page_segments_segment_type ON page_segments (segment_type);


-- =============================================
-- Junction table for page defects (Many-to-Many)
-- =============================================
CREATE TABLE IF NOT EXISTS page_defects (
    page_identifier TEXT NOT NULL,
    defect_type TEXT NOT NULL CHECK (defect_type IN (
        'skew', 'blur', 'watermark', 'stain', 'crease', 'hole_punch',
        'low_contrast', 'cut_off', 'scanned'
    )),
    -- Composite primary key ensures each defect type is listed only once per page
    PRIMARY KEY (page_identifier, defect_type),
    -- Foreign key constraint ensures page_identifier exists in the main table
    -- ON DELETE CASCADE means if a page_profile is deleted, its corresponding defects are also deleted
    CONSTRAINT fk_page_identifier
        FOREIGN KEY(page_identifier)
        REFERENCES page_profiles(page_identifier)
        ON DELETE CASCADE
);

-- Add comments
COMMENT ON TABLE page_defects IS 'Junction table mapping pages to the types of visual defects observed on them.';
COMMENT ON COLUMN page_defects.page_identifier IS 'Foreign key referencing the unique page identifier in page_profiles.';
COMMENT ON COLUMN page_defects.defect_type IS 'The type of visual defect observed on the page.';

-- Index for faster lookups based on defect type (optional)
CREATE INDEX IF NOT EXISTS idx_page_defects_defect_type ON page_defects (defect_type);


-- =============================================
-- Table to log failures during S3 to PostgreSQL upload
-- =============================================
CREATE TABLE IF NOT EXISTS upload_failures (
    failure_id BIGSERIAL PRIMARY KEY, -- Auto-incrementing primary key for each failure event
    page_identifier TEXT,             -- The page identifier (dedup_id) if available during failure
    s3_key TEXT NOT NULL,             -- The full S3 key of the JSON file that failed processing
    failure_reason TEXT,              -- Description of the error (e.g., S3 download error, JSON parse error, DB connection error, DB insert error)
    attempt_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP -- Timestamp when the upload attempt failed
);

-- Add comments
COMMENT ON TABLE upload_failures IS 'Logs failures encountered during the process of uploading extracted data from S3 to PostgreSQL.';
COMMENT ON COLUMN upload_failures.failure_id IS 'Unique identifier for the failure log entry.';
COMMENT ON COLUMN upload_failures.page_identifier IS 'The unique page identifier (dedup_id), may be NULL if failure occurred before parsing JSON.';
COMMENT ON COLUMN upload_failures.s3_key IS 'Full S3 key for the JSON file that failed to be processed and uploaded.';
COMMENT ON COLUMN upload_failures.failure_reason IS 'Details about the error that occurred during the upload attempt.';
COMMENT ON COLUMN upload_failures.attempt_timestamp IS 'Timestamp when the failure was recorded.';

-- Indexes for querying failures
CREATE INDEX IF NOT EXISTS idx_upload_failures_s3_key ON upload_failures (s3_key);
CREATE INDEX IF NOT EXISTS idx_upload_failures_timestamp ON upload_failures (attempt_timestamp);
CREATE INDEX IF NOT EXISTS idx_upload_failures_page_identifier ON upload_failures (page_identifier); -- Index if querying by page_id is common