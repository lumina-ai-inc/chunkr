-- This file should undo anything in `up.sql`

CREATE OR REPLACE FUNCTION maintain_monthly_usage_cron() RETURNS void AS $$
DECLARE
    usage_record RECORD;
BEGIN
    -- Find records where billing cycle has ended
    FOR usage_record IN 
        SELECT DISTINCT ON (user_id) 
            user_id, tier, usage_limit, billing_cycle_end
        FROM monthly_usage
        WHERE billing_cycle_end <= CURRENT_TIMESTAMP
        ORDER BY user_id, billing_cycle_end DESC
    LOOP
        -- Create new row for next billing cycle
        INSERT INTO monthly_usage (
            user_id,
            usage_type,
            usage,
            overage_usage,
            year,
            month,
            tier,
            usage_limit,
            billing_cycle_start,
            billing_cycle_end
        ) VALUES (
            usage_record.user_id,
            'Page',
            0,
            0,
            EXTRACT(YEAR FROM usage_record.billing_cycle_end),
            EXTRACT(MONTH FROM usage_record.billing_cycle_end),
            usage_record.tier,
            usage_record.usage_limit,
            usage_record.billing_cycle_end,
            usage_record.billing_cycle_end + INTERVAL '30 days'
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;