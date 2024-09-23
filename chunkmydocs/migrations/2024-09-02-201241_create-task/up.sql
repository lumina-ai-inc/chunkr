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

ALTER TABLE TASKS ADD COLUMN IF NOT EXISTS image_folder_location TEXT;

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


CREATE OR REPLACE FUNCTION validate_usage() RETURNS TRIGGER AS $$
DECLARE
    v_page_usage INTEGER;
    v_page_limit INTEGER;
    v_segment_usage INTEGER;
    v_segment_limit INTEGER;
    v_usage_type TEXT;
    v_config JSONB;
    v_user_tier TEXT;
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Get user's tier
        SELECT tier INTO v_user_tier
        FROM users
        WHERE user_id = NEW.user_id;

        -- Only proceed with validation if the user is on the 'Free' tier
        IF v_user_tier = 'Free' THEN
            -- Parse configuration string to JSONB
            v_config := NEW.configuration::JSONB;

            -- Determine usage type based on model
            IF v_config->>'model' = 'Fast' THEN
                v_usage_type := 'Fast';
            ELSIF v_config->>'model' = 'HighQuality' THEN
                v_usage_type := 'HighQuality';
            ELSE
                RAISE EXCEPTION 'Unknown model type in configuration';
            END IF;

            -- Check page count usage
            SELECT usage, usage_limit INTO v_page_usage, v_page_limit 
            FROM USAGE 
            WHERE user_id = NEW.user_id AND usage_type = v_usage_type;

            IF COALESCE(v_page_usage, 0) + NEW.page_count > COALESCE(v_page_limit, 0) THEN
                RAISE EXCEPTION 'Page usage limit exceeded';
            END IF;

            -- Check segment count usage if useVisionOCR is true
            IF v_config->>'useVisionOCR' = 'true' THEN
                SELECT usage, usage_limit INTO v_segment_usage, v_segment_limit 
                FROM USAGE 
                WHERE user_id = NEW.user_id AND usage_type = 'Segments';

                IF COALESCE(v_segment_usage, 0) + NEW.segment_count > COALESCE(v_segment_limit, 0) THEN
                    RAISE EXCEPTION 'Segment usage limit exceeded';
                END IF;
            END IF;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER validate_usage_trigger
BEFORE INSERT ON TASKS
FOR EACH ROW
EXECUTE FUNCTION validate_usage();


CREATE TRIGGER update_usage_on_success_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'Succeeded')
EXECUTE FUNCTION update_usage_on_success();





