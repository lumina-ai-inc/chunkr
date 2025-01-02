-- Your SQL goes here
CREATE TABLE IF NOT EXISTS TASKS (
    task_id TEXT PRIMARY KEY,
    user_id TEXT,
    api_key TEXT,
    file_name TEXT,
    file_size BIGINT,
    page_count INTEGER,
    segment_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    status TEXT,
    task_url TEXT,
    input_location TEXT,
    output_location TEXT,
    configuration TEXT,
    message TEXT
);

CREATE TABLE IF NOT EXISTS USAGE (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    usage INTEGER,
    usage_limit INTEGER,
    usage_type TEXT,
    unit TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION update_usage_on_success() RETURNS TRIGGER AS $$
DECLARE
    v_usage INTEGER;
    v_usage_type TEXT;
    v_segment_usage INTEGER;
    v_config JSONB;
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.status = 'Succeeded' THEN
        -- Parse configuration string to JSONB
        v_config := NEW.configuration::JSONB;

        -- Update for Fast or HighQuality
        IF v_config->>'model' = 'Fast' THEN
            v_usage_type := 'Fast';
        ELSIF v_config->>'model' = 'HighQuality' THEN
            v_usage_type := 'HighQuality';
        ELSE
            RAISE EXCEPTION 'Unknown model type in configuration';
        END IF;

        SELECT usage INTO v_usage
        FROM USAGE
        WHERE user_id = NEW.user_id AND usage_type = v_usage_type;

        UPDATE USAGE
        SET usage = COALESCE(v_usage, 0) + NEW.page_count,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = NEW.user_id AND usage_type = v_usage_type;

        -- Insert a new row if it doesn't exist
        IF NOT FOUND THEN
            INSERT INTO USAGE (user_id, usage, usage_type, unit)
            VALUES (NEW.user_id, NEW.page_count, v_usage_type, 'Page');
        END IF;

        -- Update for segments if useVisionOCR is true
        IF v_config->>'useVisionOCR' = 'true' THEN
            SELECT usage INTO v_segment_usage
            FROM USAGE
            WHERE user_id = NEW.user_id AND usage_type = 'Segment';

            UPDATE USAGE
            SET usage = COALESCE(v_segment_usage, 0) + NEW.segment_count,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = NEW.user_id AND usage_type = 'Segment';

            -- Insert a new row if it doesn't exist
            IF NOT FOUND THEN
                INSERT INTO USAGE (user_id, usage, usage_type, unit)
                VALUES (NEW.user_id, NEW.segment_count, 'Segment', 'Segment');
            END IF;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;



CREATE TRIGGER update_usage_on_success_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'Succeeded')
EXECUTE FUNCTION update_usage_on_success();





