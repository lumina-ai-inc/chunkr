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
WHERE tier IN ('PayAsYouGo');

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

INSERT INTO monthly_usage (
    user_id,
    usage_type,
    usage,
    usage_limit,
    year,
    month,
    tier,
    billing_cycle_start,
    billing_cycle_end
)
SELECT 
    u.user_id,
    'Page' as usage_type,
    0 as usage,
    200 as usage_limit,
    EXTRACT(YEAR FROM CURRENT_DATE) as year,
    EXTRACT(MONTH FROM CURRENT_DATE) as month,
    'Free' as tier,
    CURRENT_DATE as billing_cycle_start,
    (CURRENT_DATE + INTERVAL '30 days')::TIMESTAMPTZ as billing_cycle_end
FROM users u
WHERE NOT EXISTS (
    SELECT 1
    FROM monthly_usage m
    WHERE m.user_id = u.user_id
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

CREATE OR REPLACE TRIGGER trg_decrement_user_task_count
AFTER DELETE ON tasks
FOR EACH ROW
EXECUTE FUNCTION decrement_user_task_count();


-- validate usage trigger



CREATE OR REPLACE FUNCTION validate_usage() RETURNS TRIGGER AS $$
DECLARE
    v_user_tier TEXT;
    v_monthly_usage INTEGER;
    v_monthly_limit INTEGER;
    v_processing_pages INTEGER;
BEGIN
    IF NEW.page_count = OLD.page_count THEN
        RETURN NEW;
    END IF;

    SELECT tier INTO v_user_tier
    FROM users
    WHERE user_id = NEW.user_id;

    -- Only check monthly usage for Free tier
    IF v_user_tier = 'Free' THEN
        -- Lock the row and get current usage
        SELECT 
            usage, 
            usage_limit 
        INTO v_monthly_usage, v_monthly_limit
        FROM monthly_usage
        WHERE user_id = NEW.user_id 
        AND NEW.created_at >= billing_cycle_start
        AND NEW.created_at < billing_cycle_end
        FOR UPDATE;

        -- Get pages from other processing tasks
        SELECT COALESCE(SUM(page_count), 0)
        INTO v_processing_pages
        FROM tasks
        WHERE user_id = NEW.user_id
        AND status = 'Processing'
        AND created_at >= (
            SELECT billing_cycle_start 
            FROM monthly_usage 
            WHERE user_id = NEW.user_id
            AND NEW.created_at >= billing_cycle_start
            AND NEW.created_at < billing_cycle_end
        )
        AND task_id != NEW.task_id;  -- Exclude current task

        -- Check if this task would exceed limit
        IF COALESCE(v_monthly_usage, 0) + v_processing_pages + NEW.page_count > COALESCE(v_monthly_limit, 0) THEN
            RAISE EXCEPTION 'Monthly usage limit exceeded for Free tier. Current: %, Processing: %, New: %, Limit: %', 
                COALESCE(v_monthly_usage, 0), v_processing_pages, NEW.page_count, COALESCE(v_monthly_limit, 0);
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
CREATE OR REPLACE TRIGGER validate_usage_trigger
BEFORE UPDATE ON TASKS
FOR EACH ROW
EXECUTE FUNCTION validate_usage();
CREATE OR REPLACE FUNCTION update_monthly_usage() RETURNS TRIGGER AS $$
DECLARE
    old_usage INT;
    new_usage INT;
    usage_limit INT;
    old_over INT;
    new_over INT;
    partial_over INT;
    overage_rate FLOAT8;
    v_bill_date DATE;
    v_invoice_id TEXT;
BEGIN
    IF TG_OP = 'UPDATE'
       AND (OLD.status IS DISTINCT FROM NEW.status)
       AND (NEW.status = 'Succeeded' OR NEW.status = 'Failed') THEN

        SELECT mu.usage, mu.usage_limit, t.overage_rate, DATE(mu.billing_cycle_end)
        INTO old_usage, usage_limit, overage_rate, v_bill_date
        FROM monthly_usage mu
        JOIN tiers t ON t.tier = mu.tier
        WHERE mu.user_id = NEW.user_id
          AND NEW.created_at >= mu.billing_cycle_start
          AND NEW.created_at < mu.billing_cycle_end
        LIMIT 1;

        IF NEW.status = 'Failed' THEN
            new_usage := old_usage - NEW.page_count;
        ELSE
            new_usage := old_usage + NEW.page_count;
        END IF;

        old_over := GREATEST(0, old_usage - usage_limit);
        new_over := GREATEST(0, new_usage - usage_limit);
        partial_over := new_over - old_over;
        IF old_usage >= usage_limit THEN
            partial_over := NEW.page_count;
        END IF;

        UPDATE monthly_usage mu
        SET usage = new_usage,
            overage_usage = new_over,
            updated_at = CURRENT_TIMESTAMP
        WHERE mu.user_id = NEW.user_id
          AND NEW.created_at >= mu.billing_cycle_start
          AND NEW.created_at < mu.billing_cycle_end;

        IF partial_over > 0 AND NEW.status = 'Succeeded' THEN
            SELECT i.invoice_id
            INTO v_invoice_id
            FROM invoices i
            WHERE i.user_id = NEW.user_id
              AND i.invoice_status = 'Ongoing'
              AND DATE(i.bill_date) = v_bill_date
            LIMIT 1;

            IF NOT FOUND THEN
                v_invoice_id := uuid_generate_v4()::TEXT;
                INSERT INTO invoices (
                    invoice_id, user_id, tasks, invoice_status,
                    amount_due, total_pages, date_created, bill_date
                )
                VALUES (
                    v_invoice_id, NEW.user_id, ARRAY[NEW.task_id], 'Ongoing',
                    0, 0, NEW.created_at, v_bill_date
                );
            ELSE
                UPDATE invoices
                SET tasks = array_append(tasks, NEW.task_id)
                WHERE invoice_id = v_invoice_id;
            END IF;

            INSERT INTO task_invoices (
                task_id, invoice_id, usage_type, pages,
                cost, created_at, bill_date
            )
            VALUES (
                NEW.task_id, v_invoice_id, 'Page', partial_over,
                partial_over * overage_rate, NEW.created_at, v_bill_date
            );

            UPDATE invoices i
            SET amount_due = (
                SELECT SUM(ti.cost) FROM task_invoices ti WHERE ti.invoice_id = i.invoice_id
            ),
            total_pages = (
                SELECT SUM(ti.pages) FROM task_invoices ti WHERE ti.invoice_id = i.invoice_id
            )
            WHERE i.invoice_id = v_invoice_id;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

    CREATE OR REPLACE FUNCTION handle_task_invoice() RETURNS TRIGGER AS $$
    BEGIN
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER b_handle_task_invoice_trigger
    AFTER UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION handle_task_invoice();

CREATE OR REPLACE TRIGGER a_update_monthly_usage_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'Succeeded' OR NEW.status = 'Failed')
EXECUTE FUNCTION update_monthly_usage();
    
DROP TRIGGER IF EXISTS update_monthly_usage_trigger ON TASKS;
DROP FUNCTION IF EXISTS update_pre_applied_pages() CASCADE;
DROP TRIGGER IF EXISTS trigger_pre_applied_pages ON pre_applied_free_pages;
DROP TABLE IF EXISTS discounts;

CREATE OR REPLACE FUNCTION maintain_monthly_usage_cron() RETURNS void AS $$
DECLARE
    user_record RECORD;
    v_last_cycle_end TIMESTAMPTZ;
    v_next_cycle_start TIMESTAMPTZ;
    v_next_cycle_end TIMESTAMPTZ;
    v_current_date TIMESTAMPTZ;
BEGIN
    v_current_date := CURRENT_TIMESTAMP;

    -- Loop through all users
    FOR user_record IN 
        SELECT u.user_id, u.tier, t.usage_limit
        FROM users u
        JOIN tiers t ON t.tier = u.tier
    LOOP
        -- Get last billing cycle for user
        SELECT billing_cycle_end
        INTO v_last_cycle_end
        FROM monthly_usage
        WHERE user_id = user_record.user_id
        ORDER BY billing_cycle_end DESC
        LIMIT 1;

        -- If no previous cycle, start from today
        IF v_last_cycle_end IS NULL THEN
            v_next_cycle_start := date_trunc('day', v_current_date);
            v_next_cycle_end := v_next_cycle_start + INTERVAL '30 days';
            
            INSERT INTO monthly_usage (
                user_id, usage_type, usage, overage_usage,
                year, month, tier, usage_limit,
                billing_cycle_start, billing_cycle_end
            )
            VALUES (
                user_record.user_id, 'Page', 0, 0,
                EXTRACT(YEAR FROM v_next_cycle_start),
                EXTRACT(MONTH FROM v_next_cycle_start),
                user_record.tier, user_record.usage_limit,
                v_next_cycle_start, v_next_cycle_end
            );
        -- Fill any gaps between last cycle and current date
        ELSIF v_current_date >= v_last_cycle_end THEN
            v_next_cycle_start := v_last_cycle_end;
            
            WHILE v_current_date >= v_next_cycle_start LOOP
                v_next_cycle_end := v_next_cycle_start + INTERVAL '30 days';
                
                INSERT INTO monthly_usage (
                    user_id, usage_type, usage, overage_usage,
                    year, month, tier, usage_limit,
                    billing_cycle_start, billing_cycle_end
                )
                VALUES (
                    user_record.user_id, 'Page', 0, 0,
                    EXTRACT(YEAR FROM v_next_cycle_start),
                    EXTRACT(MONTH FROM v_next_cycle_start),
                    user_record.tier, user_record.usage_limit,
                    v_next_cycle_start, v_next_cycle_end
                )
                ON CONFLICT (user_id, usage_type, year, month) DO NOTHING;
                
                v_next_cycle_start := v_next_cycle_end;
            END LOOP;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Create a comment to document usage
COMMENT ON FUNCTION maintain_monthly_usage_cron() IS 'Run daily to ensure continuous monthly usage tracking for all users';


