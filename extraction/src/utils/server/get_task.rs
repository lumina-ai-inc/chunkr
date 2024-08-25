use crate::models::extraction::extract::ModelInternal;
use crate::models::extraction::task::{Status, TaskResponse};
use crate::utils::db::deadpool_postgres::{Client, Pool};
use chrono::{DateTime, Utc};
use dotenvy::dotenv;
use reqwest::Client as HttpClient;
use serde_json::json;
use std::env;

pub async fn get_task(
    pool: &Pool,
    task_id: String,
) -> Result<TaskResponse, Box<dyn std::error::Error>> {
    dotenv().ok();
    println!("Getting task with id: {}", task_id);
    let storage_url = env::var("STORAGE__URL").expect("STORAGE__URL must be set");
    let client: Client = pool.get().await?;
    let http_client = HttpClient::new();

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

    let mut file_url = None;

    let output_location: String = first_row.get("output_location");

    if task_status == Status::Succeeded {
        let download_payload = json!({
            "location": output_location,
            "expires_in": "10min"
        });

        println!("Download payload: {:?}", download_payload);

        let response = http_client
            .post(format!("{}/download", storage_url))
            .json(&download_payload)
            .send()
            .await?
            .error_for_status();

        file_url = match response {
            Ok(response) => response.text().await.ok(),
            Err(e) => {
                println!("Error downloading file: {}", e);
                return Err("Error downloading file".into());
            }
        };
    }

    Ok(TaskResponse {
        task_id,
        status: task_status,
        created_at,
        finished_at,
        message,
        file_url,
        task_url,
        expiration_time,
        model: model.to_external(),
    })
}
