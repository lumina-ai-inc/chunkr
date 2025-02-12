-- Your SQL goes here
DROP TRIGGER IF EXISTS a_update_monthly_usage_trigger ON tasks;
DROP FUNCTION IF EXISTS update_monthly_usage();
DROP TABLE IF EXISTS monthly_usage;

DROP FUNCTION IF EXISTS maintain_monthly_usage_cron();

DROP TRIGGER IF EXISTS b_handle_task_invoice_trigger ON tasks;
DROP FUNCTION IF EXISTS handle_task_invoice();


create table task_ledger (
    ledger_id text primary key,
    task_id text not null,
    user_id text not null,
    tier text not null,
    usage_type text not null,
    tier_cost float not null,
    usage_amount int not null,
    total_cost float not null,
    created_at timestamp not null);


-- migration for task ledger

INSERT INTO task_ledger (
    ledger_id,
    task_id,
    user_id,
    tier,
    usage_type,
    tier_cost,
    usage_amount,
    total_cost,
    created_at
)
SELECT 
    gen_random_uuid()::text,
    t.task_id,
    t.user_id,
    COALESCE(u.tier, 'Free'),
    'Page',
    COALESCE((SELECT price_per_month FROM tiers WHERE tier = u.tier), 0),
    COALESCE(t.page_count, 0),
    COALESCE(t.page_count, 0) * COALESCE((SELECT overage_rate FROM tiers WHERE tier = u.tier), 0),
    t.created_at
FROM tasks t
JOIN users u ON t.user_id = u.user_id
WHERE t.status = 'Succeeded'
AND t.page_count > 0;

--trigger for task ledger

CREATE OR REPLACE FUNCTION handle_task_ledger() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'Succeeded' AND NEW.page_count > 0 THEN
        INSERT INTO task_ledger (
            ledger_id,
            task_id, 
            user_id,
            tier,
            usage_type,
            tier_cost,
            usage_amount,
            total_cost,
            created_at
        )
        SELECT
            gen_random_uuid()::text,
            NEW.task_id,
            NEW.user_id,
            COALESCE(u.tier, 'Free'),
            'Page',
            COALESCE((SELECT price_per_month FROM tiers WHERE tier = u.tier), 0),
            COALESCE(NEW.page_count, 0),
            COALESCE(NEW.page_count, 0) * COALESCE((SELECT overage_rate FROM tiers WHERE tier = u.tier), 0),
            NEW.created_at
        FROM users u
        WHERE u.user_id = NEW.user_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER task_ledger_trigger  
    AFTER UPDATE ON TASKS
    FOR EACH ROW
    WHEN (NEW.status = 'Succeeded')
    EXECUTE FUNCTION handle_task_ledger();
