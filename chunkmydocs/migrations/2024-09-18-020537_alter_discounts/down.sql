-- This file should undo anything in `up.sql`
ALTER TABLE discounts
    DROP CONSTRAINT discounts_pkey,
    ADD PRIMARY KEY (user_id);
