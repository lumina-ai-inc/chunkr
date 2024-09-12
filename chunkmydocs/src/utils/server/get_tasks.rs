use crate::models::server::extract::Configuration;
use crate::models::server::task::{Status, TaskResponse};
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::storage::services::{download_to_tempfile, generate_presigned_url};
use aws_sdk_s3::Client as S3Client;
use chrono::{DateTime, Utc};
use std::io::{self, Write};

pub async fn get_tasks(
    pool: &Pool,
    s3_client: &S3Client,
    user_id: String,
    page: i64,
    limit: i64,
) -> Result<Vec<TaskResponse>, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;
    let offset = (page - 1) * limit;
    let tasks = client.query(
        "SELECT task_id, status, created_at, finished_at, expires_at, message, input_location, output_location, task_url, configuration
         FROM TASKS
         WHERE user_id = $1
         ORDER BY created_at DESC
         OFFSET $2 LIMIT $3",
        &[&user_id, &offset, &limit]
    ).await?;

    let mut task_responses = Vec::new();

    for row in tasks {
        let task_id: String = row.get("task_id");
        let expires_at: Option<DateTime<Utc>> = row.get("expires_at");
        if expires_at.is_some() && expires_at.as_ref().unwrap() < &Utc::now() {
            continue;
        }

        let status: Status = row
            .get::<_, Option<String>>("status")
            .and_then(|m| m.parse().ok())
            .ok_or("Invalid status")?;
        let created_at: DateTime<Utc> = row.get("created_at");
        let finished_at: Option<DateTime<Utc>> = row.get("finished_at");
        let message = row.get::<_, Option<String>>("message").unwrap_or_default();

        let input_location: String = row.get("input_location");
        let input_file_url = generate_presigned_url(s3_client, &input_location, None)
            .await
            .ok();
        let output_location: String = row.get("output_location");
        let mut output = None;
        if status == Status::Succeeded {
            if let Ok(temp_file) =
                download_to_tempfile(s3_client, &reqwest::Client::new(), &output_location, None)
                    .await
            {
                if let Ok(json_content) = tokio::fs::read_to_string(temp_file.path()).await {
                    output = serde_json::from_str(&json_content).ok();
                }
            }
        }

        let task_url: Option<String> = row.get("task_url");
        let configuration: Configuration = row
            .get::<_, Option<String>>("configuration")
            .and_then(|c| serde_json::from_str(&c).ok())
            .ok_or("Invalid configuration")?;

        task_responses.push(TaskResponse {
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
        });
    }

    Ok(task_responses)
}
