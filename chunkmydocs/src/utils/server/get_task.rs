use crate::models::extraction::extract::ModelInternal;
use crate::models::extraction::task::{Status, TaskResponse};
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::storage::services::generate_presigned_url;
use aws_sdk_s3::Client as S3Client;
use chrono::{DateTime, Utc};

pub async fn get_task(
    pool: &Pool,
    s3_client: &S3Client,
    task_id: String,
) -> Result<TaskResponse, Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;
    let task_and_files = client.query(
        "SELECT t.status AS task_status, t.expiration_time, t.created_at, t.finished_at, t.url AS task_url, t.model, t.message,
                f.file_id, f.status AS file_status, f.input_location, f.output_location
         FROM ingestion_tasks t
         LEFT JOIN ingestion_files f ON t.task_id = f.task_id
         WHERE t.task_id = $1",
        &[&task_id]
    ).await?;

    if task_and_files.is_empty() {
        return Err("Task not found".into());
    }

    let first_row = &task_and_files[0];
    let task_status: Status = first_row
        .get::<_, Option<String>>("task_status")
        .and_then(|m| m.parse().ok())
        .ok_or("Invalid status")?;

    let expiration_time: Option<DateTime<Utc>> = first_row.get("expiration_time");

    if expiration_time.is_some() && expiration_time.as_ref().unwrap() < &Utc::now() {
        return Err("Task expired".into());
    }

    let created_at: DateTime<Utc> = first_row.get("created_at");
    let finished_at: Option<String> = first_row.get("finished_at");
    let task_url: Option<String> = first_row.get("task_url");
    let model: ModelInternal = first_row
        .get::<_, Option<String>>("model")
        .and_then(|m| m.parse().ok())
        .ok_or("Invalid model")?;
    let message = first_row
        .get::<_, Option<String>>("message")
        .unwrap_or_default();

    let mut output_file_url = None;

    let input_location: String = first_row.get("input_location");
    let output_location: String = first_row.get("output_location");

    let input_file_url =  match generate_presigned_url(s3_client, &input_location, None).await {
        Ok(response) => Some(response),
        Err(e) => {
            println!("Error getting input file url: {}", e);
            return Err("Error getting input file url".into());
        }
    };

    if task_status == Status::Succeeded {
        output_file_url = match generate_presigned_url(s3_client, &output_location, None).await {
            Ok(response) => Some(response),
            Err(e) => {
                println!("Error getting output file url: {}", e);
                return Err("Error getting output file url".into());
            }
        };
    }

    Ok(TaskResponse {
        task_id,
        status: task_status,
        created_at,
        finished_at,
        message,
        output_file_url,
        input_file_url,
        task_url,
        expiration_time,
        model: model.to_external(),
    })
}
