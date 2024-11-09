-- Your SQL goes here
CREATE OR REPLACE FUNCTION validate_usage() RETURNS TRIGGER AS $$
DECLARE
    v_user_tier TEXT;
    v_usage_type TEXT DEFAULT 'NotKnown';
    v_config JSONB;
    v_current_year INTEGER;
    v_current_month INTEGER;
    v_usage INTEGER;
    v_limit INTEGER;
    v_lifetime_usage INTEGER;
    v_lifetime_limit INTEGER;
BEGIN
    -- Parse configuration string to JSONB
    v_config := NEW.configuration::JSONB;

    -- Determine usage type based on model
    IF v_config->>'model' = 'Fast' THEN
        v_usage_type := 'Fast';
    ELSIF v_config->>'model' = 'HighQuality' THEN
        v_usage_type := 'HighQuality';
    ELSIF v_config->>'model' = 'NoModel' THEN
        v_usage_type := 'NoModel';
    ELSE
        v_usage_type := 'NotKnown';
    END IF;

    -- Get user's tier
    SELECT tier INTO v_user_tier
    FROM users
    WHERE user_id = NEW.user_id;

    -- Different behavior based on user tier
    CASE v_user_tier
    WHEN 'Free' THEN
        -- Check lifetime usage for Free tier
        SELECT usage, usage_limit INTO v_lifetime_usage, v_lifetime_limit
        FROM USAGE
        WHERE user_id = NEW.user_id AND usage_type = v_usage_type;

        IF COALESCE(v_lifetime_usage, 0) + NEW.page_count > COALESCE(v_lifetime_limit, 0) THEN
            RAISE EXCEPTION 'Lifetime usage limit exceeded for Free tier';
        END IF;

        -- Check segment usage if useVisionOCR is true
        IF v_config->>'useVisionOCR' = 'true' THEN
            SELECT usage, usage_limit INTO v_lifetime_usage, v_lifetime_limit
            FROM USAGE
            WHERE user_id = NEW.user_id AND usage_type = 'Segment';

            IF COALESCE(v_lifetime_usage, 0) + NEW.segment_count > COALESCE(v_lifetime_limit, 0) THEN
                RAISE EXCEPTION 'Lifetime segment usage limit exceeded for Free tier';
            END IF;
        END IF;

    WHEN 'PayAsYouGo' THEN
        -- Check monthly usage for PayAsYouGo tier
        v_current_year := EXTRACT(YEAR FROM NEW.created_at);
        v_current_month := EXTRACT(MONTH FROM NEW.created_at);

        -- Get the current monthly usage
        SELECT COALESCE(SUM(usage), 0) INTO v_usage
        FROM MONTHLY_USAGE
        WHERE user_id = NEW.user_id 
          AND usage_type = v_usage_type
          AND year = v_current_year
          AND month = v_current_month;

        -- Get the monthly limit for PayAsYouGo tier
        SELECT usage_limit INTO v_limit
        FROM USAGE_LIMITS
        WHERE usage_type = v_usage_type AND tier = 'PayAsYouGo';

        IF v_usage + NEW.page_count > v_limit THEN
            RAISE EXCEPTION 'Monthly usage limit exceeded for PayAsYouGo tier';
        END IF;

        -- Check segment usage if useVisionOCR is true
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
        -- For other tiers, no usage check is performed
        -- You might want to log this or handle it differently
        RAISE NOTICE 'No usage check performed for tier: %', v_user_tier;
    END CASE;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;