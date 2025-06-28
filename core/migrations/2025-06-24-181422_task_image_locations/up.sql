-- Your SQL goes here

-- Add new columns for image locations and segment count
ALTER TABLE tasks ADD COLUMN pages_location TEXT NULL;
ALTER TABLE tasks ADD COLUMN segment_images_location TEXT NULL;
