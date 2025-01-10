ALTER TABLE TASKS 
    ALTER COLUMN expires_at DROP NOT NULL,
    ALTER COLUMN finished_at DROP NOT NULL,
    ADD COLUMN started_at TIMESTAMP WITH TIME ZONE;

UPDATE TASKS SET started_at = created_at;

CREATE OR REPLACE FUNCTION update_task_start_time()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.started_at IS NULL THEN
        NEW.started_at = CURRENT_TIMESTAMP;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_task_start_time
    BEFORE UPDATE ON TASKS
    FOR EACH ROW
    EXECUTE FUNCTION update_task_start_time();

