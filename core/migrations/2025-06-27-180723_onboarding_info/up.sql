-- Your SQL goes here
CREATE TABLE IF NOT EXISTS onboarding_records (
    id text PRIMARY KEY,
    user_id text NOT NULL,
    information jsonb NOT NULL,
    status text NOT NULL
);

-- Add foreign key constraint to reference users table
ALTER TABLE onboarding_records
ADD CONSTRAINT fk_onboarding_user_id 
FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE;

-- Add index for efficient lookups by user_id
CREATE INDEX idx_onboarding_user_id ON onboarding_records(user_id);