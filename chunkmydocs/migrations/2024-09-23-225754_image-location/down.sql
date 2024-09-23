-- This file should undo anything in `up.sql`
ALTER TABLE TASKS DROP COLUMN IF EXISTS image_folder_location;
