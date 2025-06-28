-- This file should undo anything in `up.sql`
ALTER TABLE tasks DROP COLUMN IF EXISTS pages_location;
ALTER TABLE tasks DROP COLUMN IF EXISTS segment_images_location;