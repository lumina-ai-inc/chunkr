-- Your SQL goes here
-- Drop the existing trigger
DROP TRIGGER IF EXISTS validate_usage_trigger ON TASKS;

-- Create the new trigger
CREATE TRIGGER validate_usage_trigger
BEFORE UPDATE ON TASKS
FOR EACH ROW
WHEN (NEW.page_count > 0)
EXECUTE FUNCTION validate_usage();
