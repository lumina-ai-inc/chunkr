-- Your SQL goes here
-- First update the trigger function with the fixed date logic
CREATE OR REPLACE FUNCTION update_monthly_usage() RETURNS TRIGGER AS $$
DECLARE
    usage_limit INT;
    old_over INT;
    new_over INT;
    partial_over INT;
    overage_rate FLOAT8;
    v_bill_date DATE;
    v_invoice_id TEXT;
    new_usage INT;
BEGIN
    IF TG_OP = 'UPDATE'
       AND (OLD.status IS DISTINCT FROM NEW.status)
       AND NEW.status = 'Succeeded' THEN

        -- Get billing info and recalculate total usage
        WITH task_usage AS (
            SELECT SUM(t.page_count) as total_pages
            FROM tasks t
            WHERE t.user_id = NEW.user_id
              AND t.status = 'Succeeded'
              AND t.created_at >= (
                  SELECT billing_cycle_start 
                  FROM monthly_usage 
                  WHERE user_id = NEW.user_id
                    AND NEW.created_at >= billing_cycle_start
                    AND NEW.created_at < billing_cycle_end + INTERVAL '1 day'
                  LIMIT 1
              )
              AND t.created_at < (
                  SELECT billing_cycle_end + INTERVAL '1 day'
                  FROM monthly_usage 
                  WHERE user_id = NEW.user_id
                    AND NEW.created_at >= billing_cycle_start
                    AND NEW.created_at < billing_cycle_end + INTERVAL '1 day'
                  LIMIT 1
              )
        )
        SELECT mu.usage_limit, t.overage_rate, DATE(mu.billing_cycle_end), COALESCE(tu.total_pages, 0)
        INTO usage_limit, overage_rate, v_bill_date, new_usage
        FROM monthly_usage mu
        JOIN tiers t ON t.tier = mu.tier
        LEFT JOIN task_usage tu ON true
        WHERE mu.user_id = NEW.user_id
          AND NEW.created_at >= mu.billing_cycle_start
          AND NEW.created_at < mu.billing_cycle_end + INTERVAL '1 day'
        LIMIT 1;

        -- Calculate overage
        new_over := GREATEST(0, new_usage - usage_limit);
        
        -- If they were already over limit, all new pages are overage
        -- If they weren't over limit before, only count pages that exceed the limit
        IF (new_usage - NEW.page_count) >= usage_limit THEN
            -- Already over limit, all new pages are overage
            partial_over := NEW.page_count;
        ELSE
            -- Only count pages that push them over the limit
            partial_over := GREATEST(0, new_usage - usage_limit);
        END IF;

        -- Update monthly usage
        UPDATE monthly_usage mu
        SET usage = new_usage,
            overage_usage = new_over,
            updated_at = CURRENT_TIMESTAMP
        WHERE mu.user_id = NEW.user_id
          AND NEW.created_at >= mu.billing_cycle_start
          AND NEW.created_at < mu.billing_cycle_end + INTERVAL '1 day';

        IF partial_over > 0 THEN
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

CREATE OR REPLACE TRIGGER a_update_monthly_usage_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'Succeeded')
EXECUTE FUNCTION update_monthly_usage();