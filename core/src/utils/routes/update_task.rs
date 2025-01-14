use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::task::{Configuration, Status, TaskPayload, TaskResponse};
use crate::utils::clients;
use crate::utils::services::payload::produce_extraction_payloads;
use crate::utils::services::task::get_status;
use crate::utils::storage::services::generate_presigned_url;

use chrono::Utc;
use std::error::Error;

pub async fn update_task(
    task_id: &str,
    user_id: &str,
    current_configuration: &Configuration,
) -> Result<TaskResponse, Box<dyn Error>> {
    let client = clients::get_pg_client().await?;
    let worker_config = WorkerConfig::from_env()?;

    let status = get_status(&task_id, &user_id).await?;
    match Some(status) {
        None => return Err("Task not found".into()),
        Some(status) if status != Status::Succeeded && status != Status::Failed => {
            return Err(format!("Task cannot be updated: status is {}", status).into())
        }
        _ => {}
    }

    let row = client
        .query_one(
            "SELECT configuration, input_location, output_location, image_folder_location, pdf_location, user_id, file_name 
            FROM tasks WHERE task_id = $1 AND user_id = $2",
            &[&task_id, &user_id],
        )
        .await?;

    let previous_configuration: Configuration =
        serde_json::from_str(&row.get::<_, String>("configuration"))?;
    let input_location: String = row.get("input_location");
    let output_location: String = row.get("output_location");
    let image_folder_location: String = row.get("image_folder_location");
    let pdf_location: String = row.get("pdf_location");
    let user_id: String = row.get("user_id");
    let file_name: String = row.get("file_name");
    let base_url = worker_config.server_url;
    let task_url = format!("{}/api/v1/task/{}", base_url, task_id);

    let extraction_payload = TaskPayload {
        current_configuration: current_configuration.clone(),
        file_name: file_name.clone(),
        image_folder_location,
        input_location: input_location.clone(),
        output_location,
        pdf_location,
        previous_configuration: Some(previous_configuration),
        task_id: task_id.to_string(),
        user_id: user_id.clone(),
    };

    let configuration_json = serde_json::to_string(current_configuration)?;
    client
        .execute(
            "UPDATE tasks SET configuration = $1, status = $2 WHERE task_id = $3 AND user_id = $4",
            &[
                &configuration_json,
                &Status::Starting.to_string(),
                &task_id,
                &user_id,
            ],
        )
        .await?;

    match produce_extraction_payloads(worker_config.queue_task, extraction_payload).await {
        Ok(_) => {}
        Err(e) => return Err(e),
    }

    let input_file_url = match generate_presigned_url(&input_location, true, None).await {
        Ok(response) => Some(response),
        Err(_e) => {
            return Err("Error getting input file url".into());
        }
    };

    Ok(TaskResponse {
        task_id: task_id.to_string(),
        status: Status::Starting,
        created_at: Utc::now(),
        finished_at: None,
        expires_at: None,
        output: None,
        input_file_url,
        task_url: Some(task_url),
        message: "Task queued".to_string(),
        configuration: current_configuration.clone(),
        file_name: Some(file_name.to_string()),
        page_count: None,
        pdf_url: None,
        started_at: None,
    })
}
