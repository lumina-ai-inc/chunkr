-- Your SQL goes here
CREATE TABLE IF NOT EXISTS pre_applied_free_pages (
    email TEXT,
    consumed BOOLEAN DEFAULT FALSE,
    usage_type TEXT NOT NULL,
    amount INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION update_pre_applied_pages()
RETURNS TRIGGER AS $$
BEGIN
    -- Check if user exists and tier
    IF EXISTS (
        SELECT 1 FROM users u
        WHERE u.email = NEW.email 
        AND u.tier = 'Free'
    ) THEN
        -- Add pages to existing usage limit
        UPDATE usage 
        SET usage_limit = usage_limit + NEW.amount
        FROM users u
        WHERE usage.user_id = u.user_id
        AND u.email = NEW.email
        AND usage.usage_type = NEW.usage_type;

        -- Mark as consumed
        UPDATE pre_applied_free_pages 
        SET consumed = TRUE,
            updated_at = CURRENT_TIMESTAMP
        WHERE email = NEW.email
        AND usage_type = NEW.usage_type;
    ELSIF EXISTS (
        SELECT 1 FROM users u
        WHERE u.email = NEW.email
        AND u.tier = 'PayAsYouGo'
    ) THEN
        -- Add or update discount amount
        INSERT INTO discounts (user_id, usage_type, amount)
        SELECT u.user_id, NEW.usage_type, NEW.amount
        FROM users u
        WHERE u.email = NEW.email
        ON CONFLICT (user_id, usage_type) DO UPDATE
        SET amount = discounts.amount + EXCLUDED.amount
        WHERE discounts.usage_type = EXCLUDED.usage_type;

        -- Mark as consumed
        UPDATE pre_applied_free_pages 
        SET consumed = TRUE,
            updated_at = CURRENT_TIMESTAMP
        WHERE email = NEW.email
        AND usage_type = NEW.usage_type;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_pre_applied_pages
    AFTER INSERT ON pre_applied_free_pages
    FOR EACH ROW
    EXECUTE FUNCTION update_pre_applied_pages();

