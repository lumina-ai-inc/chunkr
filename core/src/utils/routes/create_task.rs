use crate::models::auth::UserInfo;
use crate::models::task::{Configuration, Task, TaskResponse};
use crate::utils::services::payload::queue_task_payload;
use std::error::Error;
use tempfile::NamedTempFile;

pub async fn create_task(
    file: &NamedTempFile,
    file_name: Option<String>,
    user_info: &UserInfo,
    configuration: &Configuration,
) -> Result<TaskResponse, Box<dyn Error>> {
    let task = Task::new(
        user_info.user_id.as_str(),
        user_info.clone().api_key,
        configuration,
        file,
        file_name,
    )
    .await?;
    let extraction_payload = task.to_task_payload(None, None, None, None);
    queue_task_payload(extraction_payload).await?;
    let task_response: TaskResponse = task.to_task_response(false, false).await?;
    Ok(task_response)
}
