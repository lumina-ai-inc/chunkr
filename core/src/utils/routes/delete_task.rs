use crate::models::chunkr::task::Task;

pub async fn delete_task(
    task_id: String,
    user_id: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let task = match Task::get(&task_id, &user_id).await {
        Ok(task) => task,
        Err(_) => return Err("Task not found".into()),
    };
    task.delete().await?;
    Ok(())
}
