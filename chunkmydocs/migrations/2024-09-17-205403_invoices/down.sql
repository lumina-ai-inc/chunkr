-- This file should undo anything in `up.sql`
DROP TRIGGER IF EXISTS trg_handle_task_invoice ON TASKS;
DROP FUNCTION IF EXISTS handle_task_invoice();

DELETE FROM task_invoices WHERE invoice_id IN (SELECT invoice_id FROM invoices WHERE invoice_status = 'ongoing');
DELETE FROM invoices WHERE invoice_status = 'ongoing';

DROP EXTENSION IF EXISTS "uuid-ossp";

