-- Drop the indexes created in the up migration
DROP INDEX IF EXISTS idx_tasks_user_id;
DROP INDEX IF EXISTS idx_tasks_status;
DROP INDEX IF EXISTS idx_tasks_created_at;
DROP INDEX IF EXISTS idx_tasks_finished_at;
DROP INDEX IF EXISTS idx_tasks_expires_at;
