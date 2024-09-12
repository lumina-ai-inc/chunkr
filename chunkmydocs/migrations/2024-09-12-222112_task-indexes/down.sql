-- This file should undo anything in `up.sql`
DROP INDEX IF EXISTS idx_discounts_usage_type;
DROP INDEX IF EXISTS idx_task_invoices_invoice_id;
DROP INDEX IF EXISTS idx_task_invoices_usage_type;
DROP INDEX IF EXISTS idx_invoices_user_id;
DROP INDEX IF EXISTS idx_invoices_date_created;
DROP INDEX IF EXISTS idx_invoices_invoice_status;