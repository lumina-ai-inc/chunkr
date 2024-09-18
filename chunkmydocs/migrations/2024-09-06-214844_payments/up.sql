CREATE TABLE if not exists USAGE_TYPE (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    description TEXT NOT NULL,
    unit TEXT,
    cost_per_unit_dollars FLOAT
);

create table if not exists discounts(
    user_id TEXT PRIMARY KEY,
    usage_type TEXT NOT NULL,
    amount FLOAT
);

CREATE TABLE IF NOT EXISTS task_invoices (
    task_id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL,
    usage_type TEXT NOT NULL,
    pages INTEGER NOT NULL,
    cost FLOAT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    tasks TEXT[] NOT NULL,
    date_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    date_paid timestamp default null,
    invoice_status TEXT NOT NULL, --ongoing or paid or failed
    amount_due FLOAT NOT NULL,
    total_pages INTEGER NOT NULL,
    stripe_invoice_id TEXT 
);


 