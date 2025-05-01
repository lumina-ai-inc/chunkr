-- This file should undo anything in `up.sql`

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