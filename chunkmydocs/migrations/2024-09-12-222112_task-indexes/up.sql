CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON TASKS(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON TASKS(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON TASKS(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_finished_at ON TASKS(finished_at);
CREATE INDEX IF NOT EXISTS idx_tasks_expires_at ON TASKS(expires_at);