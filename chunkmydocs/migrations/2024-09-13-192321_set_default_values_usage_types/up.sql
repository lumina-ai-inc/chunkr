-- Your SQL goes here
   INSERT INTO USAGE_TYPE (id, type, description, unit, cost_per_unit_dollars)
   VALUES 
     ('Fast', 'Fast', 'Fast page processing', 'page', 0.005),
     ('HighQuality', 'HighQuality', 'High quality page processing', 'page', 0.01),
     ('Segment', 'Segment', 'Segment processing', 'segment', 0.01)
   ON CONFLICT (id) DO NOTHING;