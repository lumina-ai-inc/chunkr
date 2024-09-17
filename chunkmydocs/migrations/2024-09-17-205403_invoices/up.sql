-- Your SQL goes here
-- Function to handle invoice processing after task usage is successfully updated
CREATE OR REPLACE FUNCTION handle_task_invoice() RETURNS TRIGGER AS $$
DECLARE
    v_user_id TEXT;
    v_task_id TEXT;
    v_pages INTEGER;
    v_usage_type TEXT;
    v_created_at TIMESTAMP;
    v_invoice_id TEXT;
    v_cost_per_unit FLOAT;
    v_cost FLOAT;
BEGIN
    -- Only proceed if the task status is 'completed'
    IF NEW.status = 'completed' THEN
        v_user_id := NEW.user_id;
        v_task_id := NEW.task_id;
        v_pages := NEW.page_count;
        v_usage_type := (SELECT type FROM USAGE_TYPE WHERE id = NEW.configuration);
        v_created_at := NEW.created_at;

        -- Check if there's an ongoing invoice for this user
        SELECT invoice_id INTO v_invoice_id
        FROM invoices
        WHERE user_id = v_user_id AND invoice_status = 'ongoing'
        LIMIT 1;

        -- If no ongoing invoice, create a new one
        IF NOT FOUND THEN
            v_invoice_id := uuid_generate_v4()::TEXT;
            INSERT INTO invoices (invoice_id, user_id, tasks, invoice_status, amount_due, total_pages)
            VALUES (v_invoice_id, v_user_id, ARRAY[v_task_id], 'ongoing', 0, 0);
        ELSE
            -- Append the task_id to the existing invoice's tasks array
            UPDATE invoices
            SET tasks = array_append(tasks, v_task_id)
            WHERE invoice_id = v_invoice_id;
        END IF;

        -- Get the cost per unit for the usage type
        SELECT cost_per_unit_dollars INTO v_cost_per_unit
        FROM USAGE_TYPE
        WHERE type = v_usage_type;

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

        WHERE user_id = v_user_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to execute the handle_task_invoice function after a task is updated to 'completed'
CREATE TRIGGER trg_handle_task_invoice
AFTER UPDATE OF status ON TASKS
FOR EACH ROW
WHEN (NEW.status = 'completed')
EXECUTE FUNCTION handle_task_invoice();