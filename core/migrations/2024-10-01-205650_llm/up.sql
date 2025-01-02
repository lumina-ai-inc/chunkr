CREATE TABLE IF NOT EXISTS segment_process (
    id TEXT PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    task_id TEXT,
    segment_id TEXT,
    process_type TEXT,
    model_name TEXT,
    base_url TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    input_price FLOAT,
    output_price FLOAT,
    total_cost FLOAT,
    detail TEXT,
    latency FLOAT,
    avg_ocr_confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);