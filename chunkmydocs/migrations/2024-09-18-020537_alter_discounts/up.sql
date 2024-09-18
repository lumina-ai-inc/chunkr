-- Your SQL goes here
ALTER TABLE discounts
    DROP CONSTRAINT discounts_pkey,
    ADD PRIMARY KEY (user_id, usage_type, amount);