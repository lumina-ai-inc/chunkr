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


CREATE OR REPLACE FUNCTION maintain_monthly_usage_cron() RETURNS void AS $$
DECLARE
    user_record RECORD;
    v_last_cycle_end TIMESTAMPTZ;
    v_next_cycle_start TIMESTAMPTZ;
    v_next_cycle_end TIMESTAMPTZ;
    v_current_date TIMESTAMPTZ;
BEGIN
    v_current_date := CURRENT_TIMESTAMP;

    -- Loop through all users
    FOR user_record IN 
        SELECT u.user_id, u.tier, t.usage_limit
        FROM users u
        JOIN tiers t ON t.tier = u.tier
    LOOP
        -- Get last billing cycle for user
        SELECT billing_cycle_end
        INTO v_last_cycle_end
        FROM monthly_usage
        WHERE user_id = user_record.user_id
        ORDER BY billing_cycle_end DESC
        LIMIT 1;

        -- If no previous cycle, start from today
        IF v_last_cycle_end IS NULL THEN
            v_next_cycle_start := date_trunc('day', v_current_date);
            v_next_cycle_end := v_next_cycle_start + INTERVAL '30 days';
            
            INSERT INTO monthly_usage (
                user_id, usage_type, usage, overage_usage,
                year, month, tier, usage_limit,
                billing_cycle_start, billing_cycle_end
            )
            VALUES (
                user_record.user_id, 'Page', 0, 0,
                EXTRACT(YEAR FROM v_next_cycle_start),
                EXTRACT(MONTH FROM v_next_cycle_start),
                user_record.tier, user_record.usage_limit,
                v_next_cycle_start, v_next_cycle_end
            );
        -- Fill any gaps between last cycle and current date
        ELSIF v_current_date >= v_last_cycle_end THEN
            v_next_cycle_start := v_last_cycle_end;
            
            WHILE v_current_date >= v_next_cycle_start LOOP
                v_next_cycle_end := v_next_cycle_start + INTERVAL '30 days';
                
                INSERT INTO monthly_usage (
                    user_id, usage_type, usage, overage_usage,
                    year, month, tier, usage_limit,
                    billing_cycle_start, billing_cycle_end
                )
                VALUES (
                    user_record.user_id, 'Page', 0, 0,
                    EXTRACT(YEAR FROM v_next_cycle_start),
                    EXTRACT(MONTH FROM v_next_cycle_start),
                    user_record.tier, user_record.usage_limit,
                    v_next_cycle_start, v_next_cycle_end
                )
                ON CONFLICT (user_id, usage_type, year, month) DO NOTHING;
                
                v_next_cycle_start := v_next_cycle_end;
            END LOOP;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Create a comment to document usage
COMMENT ON FUNCTION maintain_monthly_usage_cron() IS 'Run daily to ensure continuous monthly usage tracking for all users';


