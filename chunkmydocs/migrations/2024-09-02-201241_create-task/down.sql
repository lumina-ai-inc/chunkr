-- This file should undo anything in `up.sql`
-- Drop the trigger
DROP TRIGGER IF EXISTS update_usage_on_success_trigger ON TASKS;

-- Drop the functions
DROP FUNCTION IF EXISTS update_usage();

-- Drop the tables
DROP TABLE IF EXISTS USAGE;
DROP TABLE IF EXISTS TASKS;