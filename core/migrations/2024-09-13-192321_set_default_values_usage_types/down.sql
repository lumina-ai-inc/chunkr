-- This file should undo anything in `up.sql`
   -- Remove the default values from the USAGE_TYPE table
DELETE FROM USAGE_TYPE 
WHERE id IN ('Fast', 'HighQuality', 'Segment');