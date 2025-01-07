-- This file should undo anything in `up.sql`
-- Remove the trigger
DROP TRIGGER IF EXISTS update_task_count_trigger ON tasks;

-- Remove the function
DROP FUNCTION IF EXISTS update_user_task_count();

-- Remove the task_count column from users table
ALTER TABLE users DROP COLUMN IF EXISTS task_count;
