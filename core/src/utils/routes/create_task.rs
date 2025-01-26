use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::task::{Configuration, Task, TaskResponse};
use crate::utils::services::payload::produce_extraction_payloads;
use actix_multipart::form::tempfile::TempFile;
use std::error::Error;

pub async fn create_task(
    file: &TempFile,
    user_info: &UserInfo,
    configuration: &Configuration,
) -> Result<TaskResponse, Box<dyn Error>> {
    let task = Task::new(
        user_info.user_id.as_str(),
        user_info.clone().api_key,
        configuration,
        file,
    )
    .await?;
    let extraction_payload = task.to_task_payload(None, None, None, None);
    produce_extraction_payloads(extraction_payload).await?;
    let task_response = task.to_task_response(false, false).await?;
    Ok(task_response)
}
