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

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger 
        WHERE tgname = 'update_usage_on_success_trigger'
    ) THEN
        CREATE TRIGGER update_usage_on_success_trigger
        AFTER UPDATE ON TASKS
        FOR EACH ROW
        WHEN (NEW.status = 'Succeeded')
        EXECUTE FUNCTION update_usage_on_success();
    END IF;
END $$;


CREATE OR REPLACE FUNCTION validate_usage() RETURNS TRIGGER AS $$
DECLARE
    v_user_tier TEXT;
    v_usage_type TEXT;
    v_config JSONB;
    v_current_year INTEGER;
    v_current_month INTEGER;
    v_usage INTEGER;
    v_limit INTEGER;
    v_lifetime_usage INTEGER;
    v_lifetime_limit INTEGER;
BEGIN
    v_config := NEW.configuration::JSONB;

    IF v_config->>'model' = 'Fast' THEN
        v_usage_type := 'Fast';
    ELSIF v_config->>'model' = 'HighQuality' THEN
        v_usage_type := 'HighQuality';
    ELSE
        RAISE EXCEPTION 'Unknown model type in configuration';
    END IF;

    SELECT tier INTO v_user_tier
    FROM users
    WHERE user_id = NEW.user_id;

    CASE v_user_tier
    WHEN 'Free' THEN
        SELECT usage, usage_limit INTO v_lifetime_usage, v_lifetime_limit
        FROM USAGE
        WHERE user_id = NEW.user_id AND usage_type = v_usage_type;

        IF COALESCE(v_lifetime_usage, 0) + NEW.page_count > COALESCE(v_lifetime_limit, 0) THEN
            RAISE EXCEPTION 'Lifetime usage limit exceeded for Free tier';
        END IF;

        IF v_config->>'useVisionOCR' = 'true' THEN
            SELECT usage, usage_limit INTO v_lifetime_usage, v_lifetime_limit
            FROM USAGE
            WHERE user_id = NEW.user_id AND usage_type = 'Segment';

            IF COALESCE(v_lifetime_usage, 0) + NEW.segment_count > COALESCE(v_lifetime_limit, 0) THEN
                RAISE EXCEPTION 'Lifetime segment usage limit exceeded for Free tier';
            END IF;
        END IF;

    WHEN 'PayAsYouGo' THEN
        v_current_year := EXTRACT(YEAR FROM NEW.created_at);
        v_current_month := EXTRACT(MONTH FROM NEW.created_at);

        SELECT COALESCE(SUM(usage), 0) INTO v_usage
        FROM MONTHLY_USAGE
        WHERE user_id = NEW.user_id 
          AND usage_type = v_usage_type
          AND year = v_current_year
          AND month = v_current_month;

        SELECT usage_limit INTO v_limit
        FROM USAGE_LIMITS
        WHERE usage_type = v_usage_type AND tier = 'PayAsYouGo';

        IF v_usage + NEW.page_count > v_limit THEN
            RAISE EXCEPTION 'Monthly usage limit exceeded for PayAsYouGo tier';
        END IF;

        IF v_config->>'useVisionOCR' = 'true' THEN
            SELECT COALESCE(SUM(usage), 0) INTO v_usage
            FROM MONTHLY_USAGE
            WHERE user_id = NEW.user_id 
              AND usage_type = 'Segment'
              AND year = v_current_year
              AND month = v_current_month;

            SELECT usage_limit INTO v_limit
            FROM USAGE_LIMITS
            WHERE usage_type = 'Segment' AND tier = 'PayAsYouGo';

            IF v_usage + NEW.segment_count > v_limit THEN
                RAISE EXCEPTION 'Monthly segment usage limit exceeded for PayAsYouGo tier';
            END IF;
        END IF;

    ELSE
        RAISE NOTICE 'No usage check performed for tier: %', v_user_tier;
    END CASE;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS validate_usage_trigger ON TASKS;
CREATE TRIGGER validate_usage_trigger
BEFORE INSERT ON TASKS
FOR EACH ROW
EXECUTE FUNCTION validate_usage();