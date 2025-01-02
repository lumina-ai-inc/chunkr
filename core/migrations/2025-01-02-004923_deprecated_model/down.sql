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

DROP TRIGGER IF EXISTS update_usage_on_status_change_trigger ON TASKS;


CREATE OR REPLACE FUNCTION handle_task_invoice() RETURNS TRIGGER AS $$
DECLARE
    v_user_id TEXT;
    v_task_id TEXT;
    v_pages INTEGER;
    v_segment_count INTEGER;
    v_usage_type TEXT;
    v_created_at TIMESTAMP;
    v_invoice_id TEXT;
    v_cost_per_unit FLOAT;
    v_cost FLOAT;
    v_config JSONB;
    v_current_month INTEGER;
    v_invoice_month INTEGER;
BEGIN
    -- Proceed for all users regardless of tier
    -- Only proceed if the task status is 'Succeeded'
    IF NEW.status = 'Succeeded' THEN
        v_user_id := NEW.user_id;
        v_task_id := NEW.task_id;
        v_pages := NEW.page_count;
        v_segment_count := NEW.segment_count;
        v_config := NEW.configuration::JSONB;
        v_created_at := NEW.created_at;
        v_current_month := EXTRACT(MONTH FROM v_created_at);

        -- Update for Fast, HighQuality, or Segment
        IF v_config->>'model' = 'Fast' THEN
            v_usage_type := 'Fast';
        ELSIF v_config->>'model' = 'HighQuality' THEN
            v_usage_type := 'HighQuality';
        ELSIF v_config->>'useVisionOCR' = 'true' THEN
            v_usage_type := 'Segment';
        ELSE
            RAISE EXCEPTION 'Unknown model type in configuration';
        END IF;

        -- Check if there's an ongoing invoice for this user from the current month
        SELECT invoice_id, EXTRACT(MONTH FROM date_created) INTO v_invoice_id, v_invoice_month
        FROM invoices
        WHERE user_id = v_user_id AND invoice_status = 'Ongoing'
        ORDER BY date_created DESC
        LIMIT 1;

        -- If no ongoing invoice or the last ongoing invoice is from a previous month, create a new one
        IF NOT FOUND OR v_invoice_month != v_current_month THEN
            v_invoice_id := uuid_generate_v4()::TEXT;
            INSERT INTO invoices (invoice_id, user_id, tasks, invoice_status, amount_due, total_pages, date_created)
            VALUES (v_invoice_id, v_user_id, ARRAY[v_task_id], 'Ongoing', 0, 0, v_created_at);
        ELSE
            -- Append the task_id to the existing invoice's tasks array
            UPDATE invoices
            SET tasks = array_append(tasks, v_task_id)
            WHERE invoice_id = v_invoice_id;
        END IF;

        -- Get the cost per unit for the usage type
        SELECT cost_per_unit_dollars INTO v_cost_per_unit
        FROM USAGE_TYPE
        WHERE type = v_usage_type;

        -- Calculate the cost
        IF v_usage_type = 'Segment' THEN
            v_cost := v_cost_per_unit * v_segment_count;
        ELSE
            v_cost := v_cost_per_unit * v_pages;
        END IF;

        -- Insert into task_invoices
        INSERT INTO task_invoices (task_id, invoice_id, usage_type, pages, cost, created_at)
        VALUES (v_task_id, v_invoice_id, v_usage_type, v_pages, v_cost, v_created_at);

        -- Update the invoice with the new amount_due and total_pages
        UPDATE invoices
        SET amount_due = amount_due + v_cost,
            total_pages = total_pages + v_pages
        WHERE invoice_id = v_invoice_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

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