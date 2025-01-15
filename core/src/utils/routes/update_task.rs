use crate::models::chunkr::task::{Configuration, Status, Task, TaskResponse};
use crate::utils::services::payload::produce_extraction_payloads;
use std::error::Error;

pub async fn update_task(
    previous_task: &Task,
    current_configuration: &Configuration,
) -> Result<TaskResponse, Box<dyn Error>> {
    let previous_configuration = previous_task.configuration.clone();
    let previous_status = previous_task.status.clone();
    let previous_message = previous_task.message.clone();
    let previous_version = previous_task.version.clone();

    if previous_task.status != Status::Starting && previous_task.status != Status::Failed {
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
        )
        .await?;

    produce_extraction_payloads(previous_task.to_task_payload(
        Some(previous_configuration.clone()),
        Some(previous_status),
        previous_message,
        previous_version,
    ))
    .await?;

    Ok(previous_task.to_task_response().await?)
}
