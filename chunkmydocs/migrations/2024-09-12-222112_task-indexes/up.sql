-- Your SQL goes here
-- Add index on usage_type for discounts table
CREATE INDEX idx_discounts_usage_type ON discounts (usage_type);

-- Add indexes for task_invoices table
CREATE INDEX idx_task_invoices_invoice_id ON task_invoices (invoice_id);
CREATE INDEX idx_task_invoices_usage_type ON task_invoices (usage_type);

-- Add indexes for invoices table
CREATE INDEX idx_invoices_user_id ON invoices (user_id);
CREATE INDEX idx_invoices_date_created ON invoices (date_created);
CREATE INDEX idx_invoices_invoice_status ON invoices (invoice_status);