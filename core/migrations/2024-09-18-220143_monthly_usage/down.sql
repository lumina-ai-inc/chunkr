-- This file should undo anything in `up.sql`
-- Drop the triggers
-- Drop the triggers
DROP TRIGGER IF EXISTS update_monthly_usage_trigger ON TASKS;
DROP TRIGGER IF EXISTS validate_monthly_usage_trigger ON TASKS;

-- Drop the functions
DROP FUNCTION IF EXISTS update_monthly_usage();
DROP FUNCTION IF EXISTS validate_monthly_usage();

-- Drop the tables
DROP TABLE IF EXISTS MONTHLY_USAGE;
DROP TABLE IF EXISTS USAGE_LIMITS;

-- Drop the index
DROP INDEX IF EXISTS idx_monthly_usage_user_id_type_year_month;







