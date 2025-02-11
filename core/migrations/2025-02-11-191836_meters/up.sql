-- Your SQL goes here
DROP TRIGGER IF EXISTS a_update_monthly_usage_trigger ON tasks;
DROP FUNCTION IF EXISTS update_monthly_usage();
DROP TABLE IF EXISTS monthly_usage;

