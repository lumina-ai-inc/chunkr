DROP TRIGGER IF EXISTS update_usage_on_success_trigger ON TASKS;
DROP FUNCTION IF EXISTS update_usage_on_success();
DROP TRIGGER IF EXISTS trg_handle_task_invoice ON TASKS;
DROP TRIGGER IF EXISTS update_usage_on_status_change_trigger ON TASKS;
DROP FUNCTION IF EXISTS update_usage_on_status_change();
