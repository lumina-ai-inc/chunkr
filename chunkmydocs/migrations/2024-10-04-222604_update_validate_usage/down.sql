-- This file should undo anything in `up.sql`
-- Drop the new trigger
DROP TRIGGER IF EXISTS validate_usage_trigger ON TASKS;

-- Recreate the original trigger
CREATE OR REPLACE TRIGGER validate_usage_trigger
BEFORE INSERT ON TASKS
FOR EACH ROW
EXECUTE FUNCTION validate_usage();
