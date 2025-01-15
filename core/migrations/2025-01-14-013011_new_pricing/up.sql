-- Your SQL goes here

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
    subscription_id TEXT PRIMARY KEY,
    stripe_subscription_id TEXT,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    tier TEXT NOT NULL REFERENCES tiers(tier),
    last_paid_date TIMESTAMPTZ,
    last_paid_status TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
-- Insert default tiers
INSERT INTO tiers (name, price_per_month, usage_limit, overage_rate) VALUES
('Free', 0.00, 200, 0),
('Starter', 50.00, 5000, 0.01),
('Dev', 200.00, 25000, 0.008),
('Team', 500.00, 100000, 0.005),
('SelfHosted', 0, 1000000000, 0),
('PayAsYouGo',0,0,0.01);

-- Update monthly_usage table
ALTER TABLE monthly_usage 
    ADD COLUMN overage_usage INT4 DEFAULT 0;
    ADD COLUMN tier TEXT REFERENCES tiers(tier);
    ADD COLUMN usage_limit INT4;

DROP Table USAGE_LIMITS;

ALTER TABLE usage DROP COLUMN usage_limit;


-- ---------
-- Migration
-- ---------

-- Migrate existing users to appropriate tiers
-- First identify and store PayAsYouGo users

WITH moved_users AS (
    UPDATE users 
    SET tier = 'Free'
    WHERE tier = 'PayAsYouGo'
    RETURNING user_id
)
-- Give 5000 pages to moved users and 200 to everyone else
INSERT INTO monthly_usage (user_id, usage_type, usage, usage_limit, year, month, tier)
SELECT 
    user_id,
    'Page',
    0,
    CASE WHEN user_id IN (SELECT user_id FROM moved_users) THEN 5000 ELSE 200 END,
    EXTRACT(YEAR FROM CURRENT_TIMESTAMP),
    EXTRACT(MONTH FROM CURRENT_TIMESTAMP),
    'Free'
FROM users
ON CONFLICT (user_id, usage_type, year, month)
DO UPDATE SET usage = 0;


--------------
-- TRIGGERS --
--------------

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
        AND year = v_current_year
        AND month = v_current_month;

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
    v_current_month INTEGER;
    v_invoice_month INTEGER;
    v_tier TEXT;
    v_overage_rate FLOAT8;
    v_overage_usage INTEGER;
BEGIN
    IF NEW.status = 'Succeeded' THEN
        v_user_id := NEW.user_id;
        v_task_id := NEW.task_id;
        v_pages := NEW.page_count;
        v_created_at := NEW.created_at;
        v_current_month := EXTRACT(MONTH FROM v_created_at);

        SELECT t.tier, t.overage_rate, mu.overage_usage
        INTO v_tier, v_overage_rate, v_overage_usage
        FROM tiers t
        JOIN monthly_usage mu ON mu.user_id = t.tier
        WHERE mu.user_id = v_user_id
          AND mu.usage_type = v_usage_type
          AND mu.year = EXTRACT(YEAR FROM v_created_at)
          AND mu.month = v_current_month;

        IF v_overage_usage > 0 THEN
            v_cost_per_unit := v_overage_rate;

            SELECT invoice_id, EXTRACT(MONTH FROM date_created)
            INTO v_invoice_id, v_invoice_month
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
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;



CREATE OR REPLACE FUNCTION update_monthly_usage() RETURNS TRIGGER AS $$
DECLARE
    v_usage_type TEXT := 'Page';
    v_current_year INTEGER := EXTRACT(YEAR FROM NEW.created_at);
    v_current_month INTEGER := EXTRACT(MONTH FROM NEW.created_at);
    v_usage_limit INTEGER;
    v_current_usage INTEGER;
    v_overage_usage INTEGER;
    v_tier TEXT;
BEGIN
    IF TG_OP = 'UPDATE' AND NEW.status = 'Succeeded' THEN
        SELECT t.usage_limit, s.tier
        INTO v_usage_limit, v_tier
        FROM subscriptions s
        JOIN tiers t ON s.tier = t.tier
        WHERE s.user_id = NEW.user_id;

        IF v_tier = 'PayAsYouGo' THEN
            -- Add all page counts to overage_usage
            UPDATE monthly_usage
            SET overage_usage = overage_usage + NEW.page_count,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = NEW.user_id
              AND usage_type = v_usage_type
              AND year = v_current_year
              AND month = v_current_month;

            IF NOT FOUND THEN
                INSERT INTO monthly_usage (user_id, usage, overage_usage, usage_type, year, month)
                VALUES (NEW.user_id, 0, NEW.page_count, v_usage_type, v_current_year, v_current_month);
            END IF;
        ELSE
            SELECT usage, overage_usage
            INTO v_current_usage, v_overage_usage
            FROM monthly_usage
            WHERE user_id = NEW.user_id
              AND usage_type = v_usage_type
              AND year = v_current_year
              AND month = v_current_month;

            IF v_current_usage IS NULL THEN
                INSERT INTO monthly_usage (user_id, usage, overage_usage, usage_type, year, month)
                VALUES (
                    NEW.user_id, 
                    LEAST(NEW.page_count, v_usage_limit), 
                    GREATEST(NEW.page_count - v_usage_limit, 0), 
                    v_usage_type, 
                    v_current_year, 
                    v_current_month
                );
            ELSE
                UPDATE monthly_usage
                SET usage = usage + LEAST(NEW.page_count, GREATEST(v_usage_limit - v_current_usage, 0)),
                    overage_usage = overage_usage + GREATEST(NEW.page_count - LEAST(NEW.page_count, GREATEST(v_usage_limit - v_current_usage, 0)), 0),
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = NEW.user_id
                  AND usage_type = v_usage_type
                  AND year = v_current_year
                  AND month = v_current_month;
            END IF;
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