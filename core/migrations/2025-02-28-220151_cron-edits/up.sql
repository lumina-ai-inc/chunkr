-- Your SQL goes here
CREATE OR REPLACE FUNCTION maintain_monthly_usage_cron() RETURNS void AS $$
DECLARE
    usage_record RECORD;
BEGIN
    FOR usage_record IN 
        SELECT 
            mu.user_id, 
            mu.billing_cycle_end,
            u.tier,
            t.usage_limit
        FROM (
            SELECT DISTINCT ON (user_id) *
            FROM monthly_usage
            ORDER BY user_id, updated_at DESC
        ) mu
        JOIN users u ON mu.user_id = u.user_id
        JOIN tiers t ON u.tier = t.tier
        WHERE mu.billing_cycle_end <= CURRENT_TIMESTAMP
    LOOP
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