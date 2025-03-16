use crate::models::chunkr::task::{Task, TaskQuery, TaskResponse};
use chrono::{DateTime, Utc};

pub async fn get_task(
    task_id: String,
    user_id: String,
    task_query: TaskQuery,
) -> Result<TaskResponse, Box<dyn std::error::Error>> {
    let task = match Task::get(&task_id, &user_id).await {
        Ok(task) => task,
        Err(_) => return Err("Task not found".into()),
    };
    let expires_at: Option<DateTime<Utc>> = task.expires_at;
    if expires_at.is_some() && expires_at.unwrap() < Utc::now() {
        return Err("Task expired".into());
    }
    task.to_task_response(task_query.include_chunks, task_query.base64_urls)
        .await
}
