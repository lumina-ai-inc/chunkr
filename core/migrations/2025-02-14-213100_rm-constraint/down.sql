-- This file should undo anything in `up.sql`

CREATE OR REPLACE FUNCTION update_usage_on_status_change() RETURNS TRIGGER AS $$
DECLARE
    v_usage INTEGER;
BEGIN
    SELECT usage INTO v_usage
    FROM USAGE
    WHERE user_id = NEW.user_id AND usage_type = 'Page';

    IF NEW.status != 'Failed' AND NEW.page_count > 0 AND NEW.page_count != OLD.page_count THEN
        UPDATE USAGE
        SET usage = COALESCE(v_usage, 0) + NEW.page_count,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = NEW.user_id AND usage_type = 'Page';

        IF NOT FOUND THEN
            INSERT INTO USAGE (user_id, usage, usage_type, unit)
            VALUES (NEW.user_id, NEW.page_count, 'Page', 'Page');
        END IF;
    ELSIF NEW.status = 'Failed' AND OLD.status != 'Failed' THEN
        UPDATE USAGE
        SET usage = GREATEST(COALESCE(v_usage, 0) - NEW.page_count, 0),
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = NEW.user_id AND usage_type = 'Page';
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_usage_on_status_change_trigger
AFTER UPDATE ON TASKS
FOR EACH ROW
EXECUTE FUNCTION update_usage_on_status_change();

ALTER TABLE monthly_usage
ADD CONSTRAINT monthly_usage_user_id_usage_type_year_month_key 
UNIQUE (user_id, usage_type, year, month);
