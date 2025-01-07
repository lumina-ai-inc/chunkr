-- Your SQL goes here
-- Create usage limits table
-- Create usage limits table with normalized schema
CREATE TABLE IF NOT EXISTS USAGE_LIMITS (
    id SERIAL PRIMARY KEY,
    usage_type TEXT NOT NULL,
    tier TEXT NOT NULL,
    usage_limit INTEGER NOT NULL,
    UNIQUE (usage_type, tier)
);

-- Insert default usage limits
INSERT INTO USAGE_LIMITS (usage_type, tier, usage_limit) VALUES
('Fast', 'Free', 1000),
('Fast', 'PayAsYouGo', 1000000),
('HighQuality', 'Free', 500),
('HighQuality', 'PayAsYouGo', 1000000),
('Segment', 'Free', 250),
('Segment', 'PayAsYouGo', 1000000);

-- Create monthly usage table
CREATE TABLE IF NOT EXISTS MONTHLY_USAGE (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    usage INTEGER DEFAULT 0,
    usage_type TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, usage_type, year, month)
);

-- Create index for faster queries
CREATE INDEX idx_monthly_usage_user_id_type_year_month ON MONTHLY_USAGE (user_id, usage_type, year, month);

-- Function to update monthly usage
CREATE OR REPLACE FUNCTION update_monthly_usage() RETURNS TRIGGER AS $$
DECLARE
    v_usage_type TEXT;
    v_config JSONB;
    v_current_year INTEGER;
    v_current_month INTEGER;
    v_last_usage_year INTEGER;
    v_last_usage_month INTEGER;
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.status = 'Succeeded' THEN
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

        v_current_year := EXTRACT(YEAR FROM NEW.created_at);
        v_current_month := EXTRACT(MONTH FROM NEW.created_at);

        -- Get the last usage record
        SELECT year, month INTO v_last_usage_year, v_last_usage_month
        FROM MONTHLY_USAGE
        WHERE user_id = NEW.user_id AND usage_type = v_usage_type
        ORDER BY year DESC, month DESC
        LIMIT 1;

        -- If it's a new month or there's no previous record, insert a new row
        IF v_last_usage_year IS NULL OR v_last_usage_month IS NULL OR
           (v_current_year > v_last_usage_year) OR 
           (v_current_year = v_last_usage_year AND v_current_month > v_last_usage_month) THEN
            INSERT INTO MONTHLY_USAGE (user_id, usage, usage_type, year, month)
            VALUES (NEW.user_id, NEW.page_count, v_usage_type, v_current_year, v_current_month);
        ELSE
            -- Update existing monthly usage for pages
            UPDATE MONTHLY_USAGE
            SET usage = usage + NEW.page_count,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = NEW.user_id 
              AND usage_type = v_usage_type 
              AND year = v_current_year 
              AND month = v_current_month;
        END IF;

        -- Handle segment usage if useVisionOCR is true
        IF v_config->>'useVisionOCR' = 'true' THEN
            -- Get the last segment usage record
            SELECT year, month INTO v_last_usage_year, v_last_usage_month
            FROM MONTHLY_USAGE
            WHERE user_id = NEW.user_id AND usage_type = 'Segment'
            ORDER BY year DESC, month DESC
            LIMIT 1;

            -- If it's a new month or there's no previous record, insert a new row
            IF v_last_usage_year IS NULL OR v_last_usage_month IS NULL OR
               (v_current_year > v_last_usage_year) OR 
               (v_current_year = v_last_usage_year AND v_current_month > v_last_usage_month) THEN
                INSERT INTO MONTHLY_USAGE (user_id, usage, usage_type, year, month)
                VALUES (NEW.user_id, NEW.segment_count, 'Segment', v_current_year, v_current_month);
            ELSE
                -- Update existing monthly usage for segments
                UPDATE MONTHLY_USAGE
                SET usage = usage + NEW.segment_count,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = NEW.user_id 
                  AND usage_type = 'Segment' 
                  AND year = v_current_year 
                  AND month = v_current_month;
            END IF;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE TRIGGER update_monthly_usage_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'Succeeded')
EXECUTE FUNCTION update_monthly_usage();

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