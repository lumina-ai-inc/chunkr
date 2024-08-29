-- Your SQL goes here
CREATE TABLE IF NOT EXISTS INGESTION_USAGE (
    task_id TEXT PRIMARY KEY,
    user_id TEXT,
    api_key TEXT,
    usage_type TEXT,
    usage FLOAT,
    usage_unit TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS USAGE_LIMIT (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    usage_type TEXT,
    usage_limit FLOAT,
    usage_unit TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS INGESTION_TASKS (
    task_id TEXT PRIMARY KEY,
    file_count INTEGER,
    total_size BIGINT,
    total_pages INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    finished_at TEXT,
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

-- Add a unique constraint to the api_key_usage table
ALTER TABLE public.api_key_usage
ADD CONSTRAINT api_key_usage_unique UNIQUE (api_key, usage_type, service);

-- Add a unique constraint to the api_users table
ALTER TABLE public.api_users
ADD CONSTRAINT api_users_unique UNIQUE (key, user_id);

CREATE OR REPLACE FUNCTION update_api_key_usage() RETURNS TRIGGER AS $$
DECLARE
    v_user_id TEXT;
    v_usage_type TEXT;
    v_service TEXT;
BEGIN
    IF (TG_OP = 'INSERT') OR (TG_OP = 'UPDATE' AND NEW.status = 'Succeeded') THEN
        -- Get the user_id for the given api_key
        SELECT user_id INTO v_user_id FROM public.api_keys WHERE key = NEW.api_key;

        -- Determine the usage type
        IF NEW.total_pages > 1000 THEN
            v_usage_type := 'PAID';
        ELSE
            v_usage_type := 'FREE';
        END IF;

        -- Determine the service type
        IF NEW.model = 'extraction_model' THEN
            v_service := 'EXTRACTION';
        ELSE
            v_service := 'SEARCH';
        END IF;

        -- Update api_key_usage table
        INSERT INTO public.api_key_usage (api_key, usage, usage_type, service)
        VALUES (NEW.api_key, NEW.total_pages, v_usage_type, v_service)
        ON CONFLICT (api_key, usage_type, service)
        DO UPDATE SET usage = public.api_key_usage.usage + EXCLUDED.usage;

        -- Update api_users table
        INSERT INTO public.api_users (key, user_id, usage_type, usage, service)
        VALUES (NEW.api_key, v_user_id, v_usage_type, NEW.total_pages, v_service)
        ON CONFLICT (key, user_id)
        DO UPDATE SET 
            usage = public.api_users.usage + EXCLUDED.usage,
            usage_type = EXCLUDED.usage_type,
            service = EXCLUDED.service;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update API key usage after inserting or updating an ingestion task
CREATE TRIGGER update_api_key_usage_trigger
AFTER INSERT OR UPDATE ON INGESTION_TASKS
FOR EACH ROW
EXECUTE FUNCTION update_api_key_usage();


