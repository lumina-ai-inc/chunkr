-- Your SQL goes here
-- Add bill_date to invoices and task_invoices tables
ALTER TABLE invoices
    ADD COLUMN bill_date TIMESTAMPTZ;

ALTER TABLE task_invoices
    ADD COLUMN bill_date TIMESTAMPTZ;
-- Create tiers table first
CREATE TABLE tiers (
    tier TEXT NOT NULL UNIQUE PRIMARY KEY,
    price_per_month FLOAT8 NOT NULL,
    usage_limit INT4 NOT NULL,
    overage_rate FLOAT8 NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE subscriptions (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id),
    stripe_subscription_id TEXT,
    tier TEXT NOT NULL REFERENCES tiers(tier),
    last_paid_date TIMESTAMPTZ,
    last_paid_status TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert default tiers
INSERT INTO tiers (tier, price_per_month, usage_limit, overage_rate) VALUES
('Free', 0.00, 200, 0),
('Starter', 50.00, 5000, 0.01),
('Dev', 200.00, 25000, 0.008),
('Growth', 500.00, 100000, 0.005),
('SelfHosted', 0, 1000000000, 0),
('PayAsYouGo',0,0,0.01);

-- Update monthly_usage table
ALTER TABLE monthly_usage 
    ADD COLUMN overage_usage INT4 DEFAULT 0,
    ADD COLUMN tier TEXT REFERENCES tiers(tier),
    ADD COLUMN usage_limit INT4,
    ADD COLUMN billing_cycle_start TIMESTAMPTZ,
    ADD COLUMN billing_cycle_end TIMESTAMPTZ;

DROP Table USAGE_LIMITS;

ALTER TABLE usage DROP COLUMN usage_limit;


-- ---------
-- Migration
-- ---------

-- Migrate existing users to appropriate tiers
-- First identify and store PayAsYouGo users



UPDATE users 
SET tier = 'Free'
WHERE tier IN ('PayAsYouGo', 'SelfHosted');

-- First consolidate existing data into single rows per user/month
WITH latest_usage AS (
    SELECT DISTINCT ON (user_id, year, month)
        user_id,
        'Page' as usage_type,
        0 as usage,
        200 as usage_limit,
        year,
        month,
        'Free' as tier,
        CURRENT_DATE as billing_cycle_start,
        (CURRENT_DATE + INTERVAL '30 days')::TIMESTAMPTZ as billing_cycle_end,
        updated_at
    FROM monthly_usage
    ORDER BY user_id, year, month, updated_at DESC
)
UPDATE monthly_usage m
SET 
    usage_type = 'Page',
    usage = 0,
    usage_limit = 200,
    tier = 'Free',
    billing_cycle_start = CURRENT_DATE,
    billing_cycle_end = (CURRENT_DATE + INTERVAL '30 days')::TIMESTAMPTZ
FROM latest_usage l
WHERE m.user_id = l.user_id 
AND m.year = l.year 
AND m.month = l.month
AND m.updated_at = l.updated_at;

-- Delete the duplicate rows
DELETE FROM monthly_usage m
WHERE EXISTS (
    SELECT 1
    FROM monthly_usage m2
    WHERE m2.user_id = m.user_id
    AND m2.year = m.year
    AND m2.month = m.month
    AND m2.updated_at > m.updated_at
);


--------------
-- TRIGGERS --
--------------

CREATE OR REPLACE FUNCTION decrement_user_task_count() RETURNS TRIGGER AS $$
BEGIN
    -- Decrement the task count for the user
    UPDATE users
    SET task_count = (
        SELECT COUNT(*)
        FROM tasks
        WHERE user_id = OLD.user_id
    )
    WHERE user_id = OLD.user_id;

    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_decrement_user_task_count
AFTER DELETE ON tasks
FOR EACH ROW
EXECUTE FUNCTION decrement_user_task_count();


-- validate usage trigger

-- update usage trigger


CREATE OR REPLACE FUNCTION validate_usage() RETURNS TRIGGER AS $$
DECLARE
    v_user_tier TEXT;
    v_usage_type TEXT;
    v_current_year INTEGER;
    v_current_month INTEGER;
    v_monthly_usage INTEGER;
    v_monthly_limit INTEGER;
BEGIN
    IF NEW.page_count = OLD.page_count THEN
        RETURN NEW;
    END IF;

    v_usage_type := 'Page';
    v_current_year := EXTRACT(YEAR FROM NEW.created_at);
    v_current_month := EXTRACT(MONTH FROM NEW.created_at);

    SELECT tier INTO v_user_tier
    FROM users
    WHERE user_id = NEW.user_id;

    -- Only check monthly usage for Free tier
    IF v_user_tier = 'Free' THEN
        SELECT usage, usage_limit INTO v_monthly_usage, v_monthly_limit
        FROM monthly_usage
        WHERE user_id = NEW.user_id 
        AND usage_type = v_usage_type
        AND billing_cycle_start <= NEW.created_at
        AND billing_cycle_end > NEW.created_at;

        IF v_monthly_usage IS NULL THEN
            INSERT INTO monthly_usage (
                user_id,
                usage_type,
                usage,
                usage_limit,
                billing_cycle_start,
                billing_cycle_end
            )
            SELECT 
                NEW.user_id,
                v_usage_type,
                0,
                t.usage_limit,
                date_trunc('day', NOW()),
                date_trunc('day', NOW() + interval '1 month')
            FROM tiers t
            WHERE t.tier_name = 'Free'
            RETURNING usage, usage_limit INTO v_monthly_usage, v_monthly_limit;
        END IF;

        IF COALESCE(v_monthly_usage, 0) + NEW.page_count > COALESCE(v_monthly_limit, 0) THEN
            RAISE EXCEPTION 'Monthly usage limit exceeded for Free tier';
        END IF;
    ELSIF v_user_tier NOT IN ('Free', 'SelfHosted', 'PayAsYouGo') THEN
        IF EXISTS (
            SELECT 1 FROM subscriptions 
            WHERE user_id = NEW.user_id
            AND last_paid_status = 'False'
        ) THEN
            RAISE EXCEPTION 'Usage blocked due to unpaid subscription';
        END IF;

        IF EXISTS (
            SELECT 1 FROM invoices
            WHERE user_id = NEW.user_id
            AND invoice_status NOT IN ('Paid', 'Ongoing', 'NoInvoice', 'NeedsAction')
            ORDER BY date_created DESC
            LIMIT 1
        ) THEN
            RAISE EXCEPTION 'Usage blocked due to unpaid invoice';
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS validate_usage_trigger ON TASKS;
CREATE TRIGGER validate_usage_trigger
BEFORE UPDATE ON TASKS
FOR EACH ROW
EXECUTE FUNCTION validate_usage();

CREATE OR REPLACE FUNCTION handle_task_invoice() RETURNS TRIGGER AS $$
DECLARE
    v_user_id TEXT;
    v_task_id TEXT;
    v_pages INTEGER;
    v_usage_type TEXT := 'Page';
    v_created_at TIMESTAMP;
    v_invoice_id TEXT;
    v_cost_per_unit FLOAT8;
    v_cost FLOAT;
    v_tier TEXT;
    v_overage_rate FLOAT8;
    v_overage_usage INTEGER;
    v_bill_date TIMESTAMPTZ;
BEGIN
    IF NEW.status = 'Succeeded' THEN
        v_user_id := NEW.user_id;
        v_task_id := NEW.task_id;
        v_pages := NEW.page_count;
        v_created_at := NEW.created_at;

        -- Get the current billing cycle end date (bill_date)
        SELECT billing_cycle_end INTO v_bill_date
        FROM monthly_usage
        WHERE user_id = v_user_id
          AND NEW.created_at >= billing_cycle_start 
          AND NEW.created_at < billing_cycle_end
        ORDER BY billing_cycle_start DESC
        LIMIT 1;

        SELECT t.tier, t.overage_rate, mu.overage_usage
        INTO v_tier, v_overage_rate, v_overage_usage
        FROM tiers t
        JOIN monthly_usage mu ON mu.tier = t.tier
        WHERE mu.user_id = v_user_id
          AND mu.usage_type = v_usage_type
          AND mu.billing_cycle_end = v_bill_date;

        IF v_overage_usage > 0 THEN
            v_cost_per_unit := v_overage_rate;

            -- Look for an existing invoice for this billing cycle
            SELECT invoice_id
            INTO v_invoice_id
            FROM invoices
            WHERE user_id = v_user_id 
            AND invoice_status = 'Ongoing'
            AND bill_date = v_bill_date
            LIMIT 1;

            IF NOT FOUND THEN
                v_invoice_id := uuid_generate_v4()::TEXT;
                INSERT INTO invoices (
                    invoice_id, user_id, tasks, invoice_status, 
                    amount_due, total_pages, date_created, bill_date
                )
                VALUES (
                    v_invoice_id, v_user_id, ARRAY[v_task_id], 'Ongoing',
                    0, 0, v_created_at, v_bill_date
                );
            ELSE
                UPDATE invoices
                SET tasks = array_append(tasks, v_task_id)
                WHERE invoice_id = v_invoice_id;
            END IF;

            v_cost := v_cost_per_unit * v_pages;

            INSERT INTO task_invoices (
                task_id, invoice_id, usage_type, pages, 
                cost, created_at, bill_date
            )
            VALUES (
                v_task_id, v_invoice_id, v_usage_type, v_pages,
                v_cost, v_created_at, v_bill_date
            );

            UPDATE invoices
            SET amount_due = amount_due + v_cost,
                total_pages = total_pages + v_pages
            WHERE invoice_id = v_invoice_id;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;



CREATE TRIGGER b_handle_task_invoice_trigger
    AFTER UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION handle_task_invoice();

CREATE OR REPLACE FUNCTION update_monthly_usage() RETURNS TRIGGER AS $$
DECLARE
    v_usage_type TEXT := 'Page';
    v_current_year INTEGER := EXTRACT(YEAR FROM NEW.created_at);
    v_current_month INTEGER := EXTRACT(MONTH FROM NEW.created_at);
    v_tier TEXT;
    v_usage_limit INTEGER;
    v_billing_cycle_start TIMESTAMPTZ;
    v_billing_cycle_end TIMESTAMPTZ;
    v_current_usage INTEGER;
    v_overage_usage INTEGER;
    v_new_usage INTEGER;
BEGIN
    
    IF TG_OP = 'UPDATE'
       AND (OLD.status IS DISTINCT FROM NEW.status)
       AND NEW.status = 'Succeeded' THEN
        
        RAISE NOTICE 'Processing usage update for user: %, pages: %', 
            NEW.user_id, NEW.page_count;
        
        SELECT u.tier, t.usage_limit
        INTO v_tier, v_usage_limit
        FROM users u
        JOIN tiers t ON t.tier = u.tier
        WHERE u.user_id = NEW.user_id;

        SELECT billing_cycle_start, billing_cycle_end
        INTO v_billing_cycle_start, v_billing_cycle_end
        FROM monthly_usage
        WHERE user_id = NEW.user_id
        ORDER BY billing_cycle_end DESC
        LIMIT 1;

        IF NOT FOUND THEN
            v_billing_cycle_start := date_trunc('day', NEW.created_at);
            v_billing_cycle_end := v_billing_cycle_start + INTERVAL '30 days';
        ELSE
            WHILE NEW.created_at >= v_billing_cycle_end LOOP
                v_billing_cycle_start := v_billing_cycle_end;
                v_billing_cycle_end := v_billing_cycle_start + INTERVAL '30 days';
            END LOOP;
        END IF;

        SELECT usage, overage_usage
        INTO v_current_usage, v_overage_usage
        FROM monthly_usage
        WHERE user_id = NEW.user_id
          AND billing_cycle_start = v_billing_cycle_start
          AND billing_cycle_end = v_billing_cycle_end
          AND usage_type = v_usage_type;

        IF v_current_usage IS NULL THEN
            INSERT INTO monthly_usage (
                user_id,
                usage,
                overage_usage,
                usage_type,
                year,
                month,
                tier,
                usage_limit,
                billing_cycle_start,
                billing_cycle_end
            )
            VALUES (
                NEW.user_id,
                NEW.page_count,
                0,
                v_usage_type,
                v_current_year,
                v_current_month,
                v_tier,
                v_usage_limit,
                v_billing_cycle_start,
                v_billing_cycle_end
            );
        ELSE
            v_new_usage := v_current_usage + NEW.page_count;
            UPDATE monthly_usage
            SET usage = v_new_usage,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = NEW.user_id
              AND billing_cycle_start = v_billing_cycle_start
              AND billing_cycle_end = v_billing_cycle_end
              AND usage_type = v_usage_type;
        END IF;

        UPDATE monthly_usage
        SET overage_usage = GREATEST(usage - usage_limit, 0)
        WHERE user_id = NEW.user_id
          AND billing_cycle_start = v_billing_cycle_start
          AND billing_cycle_end = v_billing_cycle_end
          AND usage_type = v_usage_type;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_monthly_usage_trigger ON TASKS;



CREATE OR REPLACE TRIGGER a_update_monthly_usage_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'Succeeded')
EXECUTE FUNCTION update_monthly_usage();
    
DROP FUNCTION IF EXISTS update_pre_applied_pages() CASCADE;
DROP TRIGGER IF EXISTS trigger_pre_applied_pages ON pre_applied_free_pages;
DROP TABLE IF EXISTS discounts;


