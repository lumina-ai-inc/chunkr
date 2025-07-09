use crate::models::auth::UserInfo;
use crate::models::task::{Configuration, Status, Task, TaskResponse};
use crate::utils::services::payload::queue_task_payload;
use std::error::Error;

pub async fn update_task(
    previous_task: &Task,
    current_configuration: &Configuration,
    user_info: &UserInfo,
) -> Result<TaskResponse, Box<dyn Error>> {
    let previous_configuration = previous_task.configuration.clone();
    let previous_status = previous_task.status.clone();
    let previous_message = previous_task.message.clone();
    let previous_version = previous_task.version.clone();

    if previous_task.status != Status::Succeeded && previous_task.status != Status::Failed {
        println!("Task cannot be updated: status is {}", previous_task.status);
        return Err(format!("Task cannot be updated: status is {}", previous_task.status).into());
    }

    let mut current_task = previous_task.clone();
    current_task
        .update(
            Some(Status::Starting),
            Some("Task queued for update".to_string()),
            Some(current_configuration.clone()),
            None,
            None,
            None,
            None,
            None,
            true,
        )
        .await?;

    queue_task_payload(previous_task.to_task_payload(
        Some(previous_configuration.clone()),
        Some(previous_status),
        previous_message,
        previous_version,
        user_info,
    ))
    .await?;

    previous_task.to_task_response(false, false).await
}
