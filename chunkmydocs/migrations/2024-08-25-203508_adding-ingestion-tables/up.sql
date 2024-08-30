-- Your SQL goes here
CREATE TABLE IF NOT EXISTS INGESTION_TASKS (
    task_id TEXT PRIMARY KEY,
    file_count INTEGER,
    total_size BIGINT,
    total_pages INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    finished_at TEXT,
    user_id TEXT,
    api_key TEXT,
    status TEXT,
    url TEXT,
    model TEXT,
    expiration_time TIMESTAMP WITH TIME ZONE,
    message TEXT,
    FOREIGN KEY (api_key) REFERENCES api_keys(key) ON DELETE CASCADE
);

-- Create INGESTION_FILES table
CREATE TABLE IF NOT EXISTS INGESTION_FILES (
    id Text,
    file_id TEXT PRIMARY KEY,
    task_id TEXT,
    file_name TEXT,
    file_size BIGINT,
    page_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    status TEXT,
    input_location TEXT,
    output_location TEXT,
    expiration_time TIMESTAMP WITH TIME ZONE,
    model TEXT,
    FOREIGN KEY (task_id) REFERENCES INGESTION_TASKS(task_id) ON DELETE CASCADE
);


CREATE OR REPLACE FUNCTION update_usage() RETURNS TRIGGER AS $$
DECLARE
    v_usage INTEGER;
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.status = 'Succeeded' THEN
        SELECT usage INTO v_usage FROM public.users WHERE user_id = NEW.user_id;
        UPDATE public.users
        SET usage = v_usage + NEW.total_pages,
        WHERE user_id = NEW.user_id AND service = 'EXTRACTION';

    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_usage_trigger
AFTER UPDATE ON INGESTION_TASKS
FOR EACH ROW
EXECUTE FUNCTION update_usage();


CREATE OR REPLACE FUNCTION validate_usage() RETURNS TRIGGER AS $$
DECLARE
    v_usage INTEGER;
    v_usage_limit INTEGER;
BEGIN
    IF TG_OP = 'INSERT' THEN
        SELECT usage, usage_limit INTO v_usage, v_usage_limit FROM public.users WHERE user_id = NEW.user_id AND service = 'EXTRACTION';
        IF v_usage + NEW.total_pages > v_usage_limit THEN
            RAISE EXCEPTION 'Usage limit exceeded';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_usage_trigger
BEFORE INSERT ON INGESTION_TASKS
FOR EACH ROW
EXECUTE FUNCTION validate_usage();

