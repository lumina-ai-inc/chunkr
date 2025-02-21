-- Your SQL goes here
drop trigger if exists update_usage_on_status_change_trigger on tasks;
drop function if exists update_usage_on_status_change();

ALTER TABLE monthly_usage
DROP CONSTRAINT monthly_usage_user_id_usage_type_year_month_key;


create table task_ledger (
    ledger_id text primary key,
    task_id text not null,
    user_id text not null,
    tier text not null,
    usage_type text not null,
    tier_cost float not null,
    usage_amount int not null,
    total_cost float not null,
    created_at timestamp not null);


-- migration for task ledger

INSERT INTO task_ledger (
    ledger_id,
    task_id,
    user_id,
    tier,
    usage_type,
    tier_cost,
    usage_amount,
    total_cost,
    created_at
)
SELECT 
    gen_random_uuid()::text,
    t.task_id,
    t.user_id,
    COALESCE(u.tier, 'Free'),
    'Page',
    COALESCE((SELECT price_per_month FROM tiers WHERE tier = u.tier), 0),
    COALESCE(t.page_count, 0),
    COALESCE(t.page_count, 0) * COALESCE((SELECT overage_rate FROM tiers WHERE tier = u.tier), 0),
    t.created_at
FROM tasks t
JOIN users u ON t.user_id = u.user_id
WHERE t.status = 'Succeeded'
AND t.page_count > 0;

--trigger for task ledger

CREATE OR REPLACE FUNCTION handle_task_ledger() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'Succeeded' AND NEW.page_count > 0 THEN
        INSERT INTO task_ledger (
            ledger_id,
            task_id, 
            user_id,
            tier,
            usage_type,
            tier_cost,
            usage_amount,
            total_cost,
            created_at
        )
        SELECT
            gen_random_uuid()::text,
            NEW.task_id,
            NEW.user_id,
            COALESCE(u.tier, 'Free'),
            'Page',
            COALESCE((SELECT price_per_month FROM tiers WHERE tier = u.tier), 0),
            COALESCE(NEW.page_count, 0),
            COALESCE(NEW.page_count, 0) * COALESCE((SELECT overage_rate FROM tiers WHERE tier = u.tier), 0),
            NEW.created_at
        FROM users u
        WHERE u.user_id = NEW.user_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER task_ledger_trigger  
    AFTER UPDATE ON TASKS
    FOR EACH ROW
    WHEN (NEW.status = 'Succeeded')
    EXECUTE FUNCTION handle_task_ledger();

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
        -- Get tier limit
        SELECT usage_limit INTO v_monthly_limit
        FROM tiers
        WHERE tier = 'Free';

        -- Get current month's usage from ledger
        SELECT COALESCE(SUM(usage_amount), 0) INTO v_monthly_usage
        FROM task_ledger
        WHERE user_id = NEW.user_id
        AND date_trunc('month', created_at) = date_trunc('month', NEW.created_at);

        -- Get pages from other processing tasks
        SELECT COALESCE(SUM(page_count), 0)
        INTO v_processing_pages
        FROM tasks
        WHERE user_id = NEW.user_id
        AND status = 'Processing'
        AND date_trunc('month', created_at) = date_trunc('month', NEW.created_at)
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

CREATE OR REPLACE TRIGGER validate_usage_trigger
BEFORE UPDATE ON TASKS
FOR EACH ROW
EXECUTE FUNCTION validate_usage();

CREATE OR REPLACE FUNCTION update_monthly_usage() RETURNS TRIGGER AS $$
DECLARE
    v_usage_limit INT;
    v_overage_rate FLOAT8;
    v_bill_date DATE;
    v_invoice_id TEXT;
    v_new_usage INT;
BEGIN
    IF TG_OP = 'UPDATE'
       AND (OLD.status IS DISTINCT FROM NEW.status)
       AND NEW.status = 'Succeeded' THEN
        -- Get current billing cycle and calculate usage only for tasks within it
        WITH monthly_cycle AS (
            SELECT 
                mu.usage_limit AS usage_limit,
                mu.tier AS tier,
                mu.billing_cycle_end AS billing_cycle_end
            FROM monthly_usage mu
            WHERE mu.user_id = NEW.user_id
            ORDER BY mu.updated_at DESC
            LIMIT 1
        ),
        current_usage AS (
            SELECT COALESCE(SUM(tl.usage_amount), 0) AS total_pages
            FROM task_ledger tl
            WHERE tl.user_id = NEW.user_id
              AND tl.task_id != NEW.task_id
        )
        SELECT 
            mc.usage_limit,
            t.overage_rate,
            DATE(mc.billing_cycle_end),
            cu.total_pages + NEW.page_count
        INTO v_usage_limit, v_overage_rate, v_bill_date, v_new_usage
        FROM monthly_cycle mc
        JOIN tiers t ON t.tier = mc.tier
        CROSS JOIN current_usage cu;

        -- Update monthly usage with new total
        UPDATE monthly_usage mu
        SET usage = v_new_usage,
            overage_usage = GREATEST(0, v_new_usage - v_usage_limit),
            updated_at = CURRENT_TIMESTAMP
        WHERE mu.user_id = NEW.user_id
        AND mu.id = (
            SELECT id FROM monthly_usage 
            WHERE user_id = NEW.user_id 
            ORDER BY updated_at DESC 
            LIMIT 1
        );

        -- Handle overage invoicing if needed
        IF v_new_usage > v_usage_limit THEN
            -- Calculate overage for just this task
            WITH monthly_cycle AS (
                SELECT 
                    mu.usage_limit AS usage_limit,
                    mu.usage - NEW.page_count AS previous_usage
                FROM monthly_usage mu
                WHERE mu.user_id = NEW.user_id
                  AND NEW.created_at >= mu.billing_cycle_start
                  AND NEW.created_at < mu.billing_cycle_end + INTERVAL '1 day'
                ORDER BY mu.billing_cycle_end DESC
                LIMIT 1
            )
            SELECT 
                CASE 
                    WHEN mc.previous_usage >= mc.usage_limit THEN NEW.page_count
                    ELSE LEAST(NEW.page_count, v_new_usage - mc.usage_limit)
                END
            INTO v_new_usage
            FROM monthly_cycle mc;

            -- Handle invoice creation/update
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
                    v_new_usage * v_overage_rate, v_new_usage, NEW.created_at, v_bill_date
                );
            ELSE
                UPDATE invoices i
                SET tasks = array_append(i.tasks, NEW.task_id),
                    amount_due = i.amount_due + (v_new_usage * v_overage_rate),
                    total_pages = i.total_pages + v_new_usage
                WHERE i.invoice_id = v_invoice_id;
            END IF;

            -- Create task invoice record
            INSERT INTO task_invoices (
                task_id, invoice_id, usage_type, pages,
                cost, created_at, bill_date
            )
            VALUES (
                NEW.task_id, v_invoice_id, 'Page', v_new_usage,
                v_new_usage * v_overage_rate, NEW.created_at, v_bill_date
            );
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE TRIGGER a_update_monthly_usage_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'Succeeded')
EXECUTE FUNCTION update_monthly_usage();

CREATE OR REPLACE FUNCTION maintain_monthly_usage_cron() RETURNS void AS $$
DECLARE
    usage_record RECORD;
BEGIN
    -- Find records where billing cycle has ended
    FOR usage_record IN 
        SELECT DISTINCT ON (user_id) 
            user_id, tier, usage_limit, billing_cycle_end
        FROM monthly_usage
        WHERE billing_cycle_end <= CURRENT_TIMESTAMP
        ORDER BY user_id, billing_cycle_end DESC
    LOOP
        -- Create new row for next billing cycle
        INSERT INTO monthly_usage (
            user_id,
            usage_type,
            usage,
            overage_usage,
            year,
            month,
            tier,
            usage_limit,
            billing_cycle_start,
            billing_cycle_end
        ) VALUES (
            usage_record.user_id,
            'Page',
            0,
            0,
            EXTRACT(YEAR FROM usage_record.billing_cycle_end),
            EXTRACT(MONTH FROM usage_record.billing_cycle_end),
            usage_record.tier,
            usage_record.usage_limit,
            usage_record.billing_cycle_end,
            usage_record.billing_cycle_end + INTERVAL '30 days'
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;