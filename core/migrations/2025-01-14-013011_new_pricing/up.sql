-- Your SQL goes here

-- Create tiers table first
CREATE TABLE tiers (
    tier TEXT NOT NULL UNIQUE PRIMARY KEY,
    monthly_price FLOAT8 NOT NULL,
    monthly_credits INT4 NOT NULL,
    overage_rate FLOAT8 NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE subscriptions (
    subscription_id TEXT PRIMARY KEY,
    stripe_subscription_id TEXT,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    tier TEXT NOT NULL REFERENCES tiers(tier),
    last_paid_date TIMESTAMPTZ,
    last_paid_status TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
-- Insert default tiers
INSERT INTO tiers (name, monthly_price, usage_limit, overage_rate) VALUES
('Free', 0.00, 100, 0),
('Starter', 50.00, 5000, 0.01),
('Dev', 200.00, 25000, 0.008),
('Team', 500.00, 100000, 0.005),
('SelfHosted', 0, 1000000000, 0);

-- Update monthly_usage table
ALTER TABLE monthly_usage 
    ADD COLUMN overage_usage INT4 DEFAULT 0;
    ADD COLUMN tier TEXT REFERENCES tiers(tier);
    ADD COLUMN usage_limit INT4;

DROP Table USAGE_LIMITS;

ALTER TABLE usage DROP COLUMN usage_limit;


-- ---------
-- Migration
-- ---------

-- Migrate existing users to appropriate tiers


-- First identify and store PayAsYouGo users
WITH moved_users AS (
    UPDATE users 
    SET tier = 'Free'
    WHERE tier = 'PayAsYouGo'
    RETURNING user_id
)
-- Give 5000 pages to moved users and 200 to everyone else
INSERT INTO monthly_usage (user_id, usage_type, usage, usage_limit, year, month, tier)
SELECT 
    user_id,
    'Page',
    0,
    CASE WHEN user_id IN (SELECT user_id FROM moved_users) THEN 5000 ELSE 200 END,
    EXTRACT(YEAR FROM CURRENT_TIMESTAMP),
    EXTRACT(MONTH FROM CURRENT_TIMESTAMP),
    'Free'
FROM users
ON CONFLICT (user_id, usage_type, year, month)
DO UPDATE SET usage = 0;

