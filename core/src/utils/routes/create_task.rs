use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::task::{Configuration, Status, TaskPayload, TaskResponse};
use crate::utils::clients;
use crate::utils::services::payload::produce_extraction_payloads;
use crate::utils::storage::services::{generate_presigned_url, upload_to_s3};
use actix_multipart::form::tempfile::TempFile;
use chrono::Utc;
use std::{
    error::Error,
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub async fn create_task(
    file: &TempFile,
    user_info: &UserInfo,
    configuration: &Configuration,
) -> Result<TaskResponse, Box<dyn Error>> {
    let task_id = Uuid::new_v4().to_string();
    let user_id = user_info.user_id.clone();
    let api_key = user_info.api_key.clone();
    let created_at = Utc::now();
    let client = clients::get_pg_client().await?;
    let worker_config = WorkerConfig::from_env()?;
    let bucket_name = worker_config.s3_bucket;
    let base_url = worker_config.server_url;
    let task_url = format!("{}/api/v1/task/{}", base_url, task_id);
    let page_count: i32 = 0;

    let file_path: PathBuf = PathBuf::from(file.file.path());
    let file_size = file.size;
    let file_name = file.file_name.as_deref().unwrap_or("unknown.pdf");
    let path = Path::new(file_name);
    let file_stem = path
        .file_stem()
        .and_then(|s: &std::ffi::OsStr| s.to_str())
        .unwrap_or("unknown");
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("pdf");

    let input_location = format!("s3://{}/{}/{}/{}", bucket_name, user_id, task_id, file_name);
    let pdf_location = format!(
        "s3://{}/{}/{}/{}.pdf",
        bucket_name, user_id, task_id, file_stem
    );
    let output_location = format!(
        "s3://{}/{}/{}/{}.{}",
        bucket_name, user_id, task_id, file_stem, "json"
    );

    let image_folder_location =
        format!("s3://{}/{}/{}/{}", bucket_name, user_id, task_id, "images");

    let message = "Task queued".to_string();

    match upload_to_s3(&input_location, &file_path).await {
        Ok(_) => {
            let configuration_json = serde_json::to_string(configuration)?;

            match client
                .execute(
                    "INSERT INTO TASKS (
                    task_id, user_id, api_key, file_name, file_size, 
                    page_count, segment_count,
                    status, task_url, input_location, output_location, image_folder_location,
                    configuration, message, pdf_location, input_file_type
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                ) ON CONFLICT (task_id) DO NOTHING",
                    &[
                        &task_id,
                        &user_id,
                        &api_key,
                        &file_name,
                        &(file_size as i64),
                        &page_count,
                        &0i32,
                        &Status::Starting.to_string(),
                        &task_url,
                        &input_location,
                        &output_location,
                        &image_folder_location,
                        &configuration_json,
                        &message,
                        &pdf_location,
                        &extension,
                    ],
                )
                .await
            {
                Ok(_) => {}
                Err(e) => {
                    if e.to_string().contains("usage limit exceeded") {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "429 Rate Limit Error: Usage limit exceeded",
                        )));
                    } else {
                        return Err(Box::new(e) as Box<dyn Error>);
                    }
                }
            }

            let extraction_payload = TaskPayload {
                current_configuration: configuration.clone(),
                file_name: file_name.to_string(),
                image_folder_location,
                input_location: input_location.clone(),
                output_location,
                pdf_location: pdf_location.clone(),
                previous_configuration: None,
                task_id: task_id.clone(),
                user_id: user_id.clone(),
            };

            match produce_extraction_payloads(worker_config.queue_task, extraction_payload).await {
                Ok(_) => {}
                Err(e) => {
                    match client
                        .execute("DELETE FROM TASKS WHERE task_id = $1", &[&task_id])
                        .await
                    {
                        Ok(_) => {}
                        Err(e) => {
                            println!("Error deleting task from database: {:?}", e);
                        }
                    }
                    return Err(e);
                }
            }

            let input_file_url = match generate_presigned_url(&input_location, true, None).await {
                Ok(response) => Some(response),
                Err(_e) => {
                    return Err("Error getting input file url".into());
                }
            };

            Ok(TaskResponse {
                task_id: task_id.clone(),
                status: Status::Starting,
                created_at,
                finished_at: None,
                expires_at: None,
                output: None,
                input_file_url,
                task_url: Some(task_url),
                message,
                configuration: configuration.clone(),
                file_name: Some(file_name.to_string()),
                page_count: Some(page_count),
                pdf_url: None,
                started_at: None,
            })
        }
        Err(e) => Err(e),
    }
}
