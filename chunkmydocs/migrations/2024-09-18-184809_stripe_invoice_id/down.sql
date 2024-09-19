-- This file should undo anything in `up.sql`
ALTER TABLE invoices
DROP COLUMN stripe_invoice_id;
