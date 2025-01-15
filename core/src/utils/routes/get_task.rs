use crate::models::chunkr::task::{Task, TaskResponse};
use chrono::{DateTime, Utc};

pub async fn get_task(
    task_id: String,
    user_id: String,
) -> Result<TaskResponse, Box<dyn std::error::Error>> {
    let task = match Task::get_by_id(&task_id, &user_id).await {
        Ok(task) => task,
        Err(_) => return Err("Task not found".into()),
    };
    let expires_at: Option<DateTime<Utc>> = task.expires_at;
    if expires_at.is_some() && expires_at.unwrap() < Utc::now() {
        return Err("Task expired".into());
    }
    Ok(task.to_task_response().await?)
}
