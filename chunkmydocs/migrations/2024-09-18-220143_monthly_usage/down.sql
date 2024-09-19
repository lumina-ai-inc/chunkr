-- This file should undo anything in `up.sql`
-- Drop the triggers
-- Drop the triggers
DROP TRIGGER IF EXISTS update_monthly_usage_trigger ON TASKS;
DROP TRIGGER IF EXISTS validate_monthly_usage_trigger ON TASKS;

-- Drop the functions
DROP FUNCTION IF EXISTS update_monthly_usage();
DROP FUNCTION IF EXISTS validate_monthly_usage();

-- Drop the tables
DROP TABLE IF EXISTS MONTHLY_USAGE;
DROP TABLE IF EXISTS USAGE_LIMITS;

-- Drop the index
DROP INDEX IF EXISTS idx_monthly_usage_user_id_type_year_month;

-- Restore the old validate_usage function and trigger
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





