DROP TRIGGER IF EXISTS update_usage_on_success_trigger ON TASKS;

CREATE OR REPLACE FUNCTION update_usage_on_status_change() RETURNS TRIGGER AS $$
DECLARE
    v_usage INTEGER;
BEGIN
    SELECT usage INTO v_usage
    FROM USAGE
    WHERE user_id = NEW.user_id AND usage_type = 'Page';

    IF NEW.status != 'Failed' AND NEW.page_count > 0 AND NEW.page_count != OLD.page_count THEN
        UPDATE USAGE
        SET usage = COALESCE(v_usage, 0) + NEW.page_count,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = NEW.user_id AND usage_type = 'Page';

        IF NOT FOUND THEN
            INSERT INTO USAGE (user_id, usage, usage_type, unit)
            VALUES (NEW.user_id, NEW.page_count, 'Page', 'Page');
        END IF;
    ELSIF NEW.status = 'Failed' AND OLD.status != 'Failed' THEN
        UPDATE USAGE
        SET usage = GREATEST(COALESCE(v_usage, 0) - NEW.page_count, 0),
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = NEW.user_id AND usage_type = 'Page';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_usage_on_status_change_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
EXECUTE FUNCTION update_usage_on_status_change();


CREATE OR REPLACE FUNCTION validate_usage() RETURNS TRIGGER AS $$
DECLARE
    v_usage INTEGER;
    v_limit INTEGER;
BEGIN
    SELECT usage, usage_limit INTO v_usage, v_limit
    FROM USAGE
    WHERE user_id = NEW.user_id AND usage_type = 'Page';

    IF COALESCE(v_usage, 0) + NEW.page_count > COALESCE(v_limit, 0) THEN
        RAISE EXCEPTION 'Page usage limit exceeded';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


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
    v_usage_type := 'Page';
    
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

    ELSE
        RAISE NOTICE 'No usage check performed for tier: %', v_user_tier;
    END CASE;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS validate_usage_trigger ON TASKS;
CREATE TRIGGER validate_usage_trigger
BEFORE INSERT OR UPDATE ON TASKS
FOR EACH ROW
WHEN (
    (TG_OP = 'INSERT') OR 
    (TG_OP = 'UPDATE' AND NEW.page_count != OLD.page_count)
)
EXECUTE FUNCTION validate_usage();

INSERT INTO USAGE (user_id, usage, usage_limit, usage_type, unit)
SELECT 
    COALESCE(fast.user_id, hq.user_id),
    COALESCE(fast.usage, 0) + COALESCE(hq.usage, 0),
    COALESCE(fast.usage_limit, 0) + COALESCE(hq.usage_limit, 0),
    'Page',
    'Page'
FROM (SELECT user_id, usage, usage_limit 
      FROM USAGE 
      WHERE usage_type = 'Fast') fast
FULL OUTER JOIN (SELECT user_id, usage, usage_limit 
                 FROM USAGE 
                 WHERE usage_type = 'HighQuality') hq
    ON fast.user_id = hq.user_id
WHERE NOT EXISTS (
    SELECT 1 
    FROM USAGE u 
    WHERE u.user_id = COALESCE(fast.user_id, hq.user_id) 
        AND u.usage_type = 'Page'
);

DELETE FROM USAGE WHERE usage_type IN ('Fast', 'HighQuality');