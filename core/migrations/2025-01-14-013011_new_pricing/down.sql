-- Restore old usage_limits table
CREATE TABLE usage_limits (
    user_id TEXT NOT NULL,
    usage_type TEXT NOT NULL,
    usage_limit INTEGER NOT NULL,
    PRIMARY KEY (user_id, usage_type)
);

-- Restore usage_limit column
ALTER TABLE usage ADD COLUMN usage_limit INTEGER;

-- Remove new columns from monthly_usage
ALTER TABLE monthly_usage 
    DROP COLUMN overage_usage,
    DROP COLUMN tier,
    DROP COLUMN usage_limit;

-- Restore old functions
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

        SELECT invoice_id, EXTRACT(MONTH FROM date_created) INTO v_invoice_id, v_invoice_month
        FROM invoices
        WHERE user_id = v_user_id AND invoice_status = 'Ongoing'
        ORDER BY date_created DESC
        LIMIT 1;

        IF NOT FOUND OR v_invoice_month != v_current_month THEN
            v_invoice_id := uuid_generate_v4()::TEXT;
            INSERT INTO invoices (invoice_id, user_id, tasks, invoice_status, amount_due, total_pages, date_created)
            VALUES (v_invoice_id, v_user_id, ARRAY[v_task_id], 'Ongoing', 0, 0, v_created_at);
        ELSE
            UPDATE invoices
            SET tasks = array_append(tasks, v_task_id)
            WHERE invoice_id = v_invoice_id;
        END IF;

        v_cost := v_cost_per_unit * v_pages;

        INSERT INTO task_invoices (task_id, invoice_id, usage_type, pages, cost, created_at)
        VALUES (v_task_id, v_invoice_id, v_usage_type, v_pages, v_cost, v_created_at);

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
        v_config := NEW.configuration::JSONB;
        v_usage_type := 'Page';
        
        v_current_year := EXTRACT(YEAR FROM NEW.created_at);
        v_current_month := EXTRACT(MONTH FROM NEW.created_at);

        SELECT year, month INTO v_last_usage_year, v_last_usage_month
        FROM MONTHLY_USAGE
        WHERE user_id = NEW.user_id AND usage_type = v_usage_type
        ORDER BY year DESC, month DESC
        LIMIT 1;

        IF v_last_usage_year IS NULL OR v_last_usage_month IS NULL OR
           (v_current_year > v_last_usage_year) OR 
           (v_current_year = v_last_usage_year AND v_current_month > v_last_usage_month) THEN
            INSERT INTO MONTHLY_USAGE (user_id, usage, usage_type, year, month)
            VALUES (NEW.user_id, NEW.page_count, v_usage_type, v_current_year, v_current_month);
        ELSE
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