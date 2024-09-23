-- Your SQL goes here
ALTER TABLE users
ADD COLUMN task_count INTEGER DEFAULT NULL;

-- Initially calculate task count for each user
UPDATE users
SET task_count = (
    SELECT COUNT(*)
    FROM tasks
    WHERE tasks.user_id = users.user_id
);


-- Create a function to update task_count
CREATE OR REPLACE FUNCTION update_user_task_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users
    SET task_count = (
        SELECT COUNT(*)
        FROM tasks
        WHERE tasks.user_id = NEW.user_id
    )
    WHERE users.user_id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to call the function when a new task is inserted
CREATE TRIGGER update_task_count_trigger
AFTER INSERT ON tasks
FOR EACH ROW
EXECUTE FUNCTION update_user_task_count();
