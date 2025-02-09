-- This file should undo anything in `up.sql`
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

CREATE OR REPLACE TRIGGER a_update_monthly_usage_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'Succeeded' OR NEW.status = 'Failed')
EXECUTE FUNCTION update_monthly_usage();