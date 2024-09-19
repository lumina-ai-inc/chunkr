ALTER TABLE discounts
    DROP CONSTRAINT IF EXISTS discounts_pkey,
    ADD PRIMARY KEY (user_id, usage_type);