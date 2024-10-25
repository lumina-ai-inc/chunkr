-- This file should undo anything in `up.sql`
DROP TABLE IF EXISTS pre_applied_free_pages;
DROP TRIGGER IF EXISTS trigger_pre_applied_pages ON usage;
DROP FUNCTION IF EXISTS update_pre_applied_pages;