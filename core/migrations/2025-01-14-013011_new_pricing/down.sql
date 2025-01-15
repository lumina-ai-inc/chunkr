-- This file should undo anything in `up.sql`


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
    if NEW.page_count = OLD.page_count THEN
        RETURN NEW;
    END IF;

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
BEFORE UPDATE ON TASKS
FOR EACH ROW
EXECUTE FUNCTION validate_usage();

INSERT INTO USAGE (user_id, usage, usage_limit, usage_type, unit)
SELECT 
    COALESCE(fast.user_id, hq.user_id),
    LEAST(2147483647, COALESCE(fast.usage, 0) + COALESCE(hq.usage, 0)),
    LEAST(2147483647, 
        LEAST(COALESCE(fast.usage_limit, 0), 2147483647/2) + 
        LEAST(COALESCE(hq.usage_limit, 0), 2147483647/2)
    ),
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
        v_usage_type := 'Page';
        v_cost_per_unit := 0.01;

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
        -- SELECT cost_per_unit_dollars INTO v_cost_per_unit
        -- FROM USAGE_TYPE
        -- WHERE type = v_usage_type;

        -- Calculate the cost
        v_cost := v_cost_per_unit * v_pages;

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
        v_usage_type := 'Page';
        
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
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- Your SQL goes here
CREATE TABLE IF NOT EXISTS pre_applied_free_pages (
    email TEXT,
    consumed BOOLEAN DEFAULT FALSE,
    usage_type TEXT NOT NULL,
    amount INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION update_pre_applied_pages()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if user exists and tier
    IF EXISTS (
        SELECT 1 FROM users u
        WHERE u.email = NEW.email 
        AND u.tier = 'Free'
    ) THEN
        -- Add pages to existing usage limit
        UPDATE usage 
        SET usage_limit = usage_limit + NEW.amount
        FROM users u
        WHERE usage.user_id = u.user_id
        AND u.email = NEW.email
        AND usage.usage_type = NEW.usage_type;

        -- Mark as consumed
        UPDATE pre_applied_free_pages 
        SET consumed = TRUE,
            updated_at = CURRENT_TIMESTAMP
        WHERE email = NEW.email
        AND usage_type = NEW.usage_type;
    ELSIF EXISTS (
        SELECT 1 FROM users u
        WHERE u.email = NEW.email
        AND u.tier = 'PayAsYouGo'
    ) THEN
        -- Add or update discount amount
        INSERT INTO discounts (user_id, usage_type, amount)
        SELECT u.user_id, NEW.usage_type, NEW.amount
        FROM users u
        WHERE u.email = NEW.email
        ON CONFLICT (user_id, usage_type) DO UPDATE
        SET amount = discounts.amount + EXCLUDED.amount
        WHERE discounts.usage_type = EXCLUDED.usage_type;

        -- Mark as consumed
        UPDATE pre_applied_free_pages 
        SET consumed = TRUE,
            updated_at = CURRENT_TIMESTAMP
        WHERE email = NEW.email
        AND usage_type = NEW.usage_type;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_pre_applied_pages
    AFTER INSERT ON pre_applied_free_pages
    FOR EACH ROW
    EXECUTE FUNCTION update_pre_applied_pages();