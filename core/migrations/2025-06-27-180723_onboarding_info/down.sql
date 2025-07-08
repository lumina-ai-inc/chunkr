-- This file should undo anything in `up.sql`

-- Drop the index first
DROP INDEX IF EXISTS idx_onboarding_info_id;

-- Drop the table
DROP TABLE IF EXISTS onboarding_info;

-- Drop the enum type
DROP TYPE IF EXISTS onboarding_status;
