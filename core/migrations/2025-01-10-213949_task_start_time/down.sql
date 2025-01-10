DROP TRIGGER IF EXISTS set_task_start_time ON TASKS;
DROP FUNCTION IF EXISTS update_task_start_time();

ALTER TABLE TASKS 
    DROP COLUMN started_at;
