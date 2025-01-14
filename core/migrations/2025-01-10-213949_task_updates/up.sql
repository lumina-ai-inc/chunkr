ALTER TABLE TASKS 
    ADD COLUMN mime_type VARCHAR(255),
    ADD COLUMN started_at TIMESTAMP WITH TIME ZONE;

UPDATE TASKS SET started_at = created_at;

DROP TRIGGER IF EXISTS trg_handle_task_invoice ON TASKS;
DROP FUNCTION IF EXISTS handle_task_invoice();
