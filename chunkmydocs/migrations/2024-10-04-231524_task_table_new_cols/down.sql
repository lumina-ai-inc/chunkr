-- This file should undo anything in `up.sql`
ALTER TABLE tasks
DROP COLUMN pdf_location,
DROP COLUMN input_file_type;
