-- This file should undo anything in `up.sql`
-- Your SQL goes here
CREATE OR REPLACE FUNCTION update_monthly_usage() RETURNS TRIGGER AS $$
DECLARE
  rec_cycle     RECORD;
  total_usage   INT;
  overage_pages INT;
  overage_rate  FLOAT8;
  bill_date     DATE;
  inv_id        TEXT;
BEGIN
  IF TG_OP = 'UPDATE'
     AND OLD.status IS DISTINCT FROM NEW.status
     AND NEW.status = 'Succeeded'
  THEN
    -- 1) Find the one “current” cycle row by billing dates + row creation
    SELECT
      id,
      billing_cycle_start,
      billing_cycle_end,
      usage_limit,
      tier,
      created_at AS row_created_at
    INTO rec_cycle
    FROM monthly_usage
    WHERE user_id = NEW.user_id
    ORDER BY
      billing_cycle_start DESC,
      created_at          DESC
    LIMIT 1;

    -- 2) Re-sum only ledger entries in [row_created_at, billing_cycle_end)
    SELECT COALESCE(SUM(usage_amount), 0)
    INTO total_usage
    FROM task_ledger
    WHERE user_id    = NEW.user_id
      AND created_at >= rec_cycle.row_created_at
      AND created_at <  rec_cycle.billing_cycle_end;

    -- 3) Update that exact row
    UPDATE monthly_usage
    SET
      usage         = total_usage,
      overage_usage = GREATEST(total_usage - rec_cycle.usage_limit, 0),
      updated_at    = NOW()
    WHERE id = rec_cycle.id;

    -- 4) If there is any overage, invoice just the overage pages
    IF total_usage > rec_cycle.usage_limit THEN
      overage_pages := LEAST(
        NEW.page_count,
        total_usage - rec_cycle.usage_limit
      );

      -- lookup the rate and bill date
      SELECT overage_rate INTO overage_rate
        FROM tiers
        WHERE tier = rec_cycle.tier;
      bill_date := rec_cycle.billing_cycle_end::DATE;

      -- find or create the Ongoing invoice for this cycle
      SELECT invoice_id
      INTO inv_id
      FROM invoices
      WHERE user_id       = NEW.user_id
        AND invoice_status = 'Ongoing'
        AND DATE(bill_date) = bill_date
      LIMIT 1;

      IF NOT FOUND THEN
        inv_id := uuid_generate_v4()::TEXT;
        INSERT INTO invoices (
          invoice_id, user_id, tasks, invoice_status,
          amount_due, total_pages, date_created, bill_date
        ) VALUES (
          inv_id, NEW.user_id, ARRAY[NEW.task_id], 'Ongoing',
          overage_pages * overage_rate,
          overage_pages,
          NOW(),
          bill_date
        );
      ELSE
        UPDATE invoices
        SET
          tasks       = array_append(tasks, NEW.task_id),
          amount_due  = amount_due + (overage_pages * overage_rate),
          total_pages = total_pages + overage_pages
        WHERE invoice_id = inv_id;
      END IF;

      -- record in task_invoices
      INSERT INTO task_invoices (
        task_id, invoice_id, usage_type, pages,
        cost, created_at, bill_date
      ) VALUES (
        NEW.task_id, inv_id, 'Page',
        overage_pages,
        overage_pages * overage_rate,
        NOW(),
        bill_date
      );
    END IF;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER a_update_monthly_usage_trigger
AFTER UPDATE ON tasks
FOR EACH ROW
WHEN (NEW.status = 'Succeeded')
EXECUTE FUNCTION update_monthly_usage();
