use crate::models::server::extract::Configuration;
use crate::models::server::task::{Status, TaskResponse};
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::storage::services::{download_to_tempfile, generate_presigned_url};
use aws_sdk_s3::Client as S3Client;
use chrono::{DateTime, Utc};
use reqwest;
use serde_json;

pub async fn get_task(
    pool: &Pool,
    s3_client: &S3Client,
    task_id: String,
    user_id: String,
) -> Result<TaskResponse, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;
    let task_and_files = client.query(
        "SELECT status, created_at, finished_at, expires_at, message, input_location, output_location, task_url, configuration
         FROM TASKS
         WHERE task_id = $1 AND user_id = $2",
        &[&task_id, &user_id]
    ).await?;

    if task_and_files.is_empty() {
        return Err("Task not found".into());
    }

    let first_row = &task_and_files[0];

    let expires_at: Option<DateTime<Utc>> = first_row.get("expires_at");
    if expires_at.is_some() && expires_at.as_ref().unwrap() < &Utc::now() {
        return Err("Task expired".into());
    }

    let status: Status = first_row
        .get::<_, Option<String>>("status")
        .and_then(|m| m.parse().ok())
        .ok_or("Invalid status")?;
    let created_at: DateTime<Utc> = first_row.get("created_at");
    let finished_at: Option<DateTime<Utc>> = first_row.get("finished_at");
    let message = first_row
        .get::<_, Option<String>>("message")
        .unwrap_or_default();

    let input_location: String = first_row.get("input_location");
    let input_file_url = match generate_presigned_url(s3_client, &input_location, None).await {
        Ok(response) => {
            println!("Successfully generated input file URL");
            Some(response)
        }
        Err(e) => {
            println!("Error getting input file url: {}", e);
            return Err("Error getting input file url".into());
        }
    };

    let output_location: String = first_row.get("output_location");
    let mut output = None;
    if status == Status::Succeeded {
        // Download the JSON file
        let temp_file =
            download_to_tempfile(s3_client, &reqwest::Client::new(), &output_location, None)
                .await?;
        let json_content = tokio::fs::read_to_string(temp_file.path()).await?;
        // Parse the JSON content
        output = Some(serde_json::from_str(&json_content)?);
    }

    let task_url: Option<String> = first_row.get("task_url");
    let configuration: Configuration = first_row
        .get::<_, Option<String>>("configuration")
        .and_then(|c| serde_json::from_str(&c).ok())
        .ok_or("Invalid configuration")?;

    Ok(TaskResponse {
        task_id,
        status,
        created_at,
        finished_at,
        expires_at,
        message,
        output,
        input_file_url,
        task_url,
        configuration,
    })
}
