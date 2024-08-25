use actix_multipart::form::tempfile::TempFile;
use chrono::{DateTime, Utc};
use dotenvy::dotenv;
use humantime;
use lopdf::Document;
use models::ingestion::extraction::ExtractionPayload;
use models::ingestion::task::{FileResponse, Status, TaskResponse};
use models::rrq::produce::ProducePayload;
use shared::deadpool_postgres::{Client, Pool};
use shared::rrq::service::produce;
use shared::storage_service::services::upload_to_s3;
use std::env;
use std::error::Error;
use std::time::Duration;
use uuid::Uuid;

fn is_valid_pdf(buffer: &[u8]) -> Result<bool, lopdf::Error> {
    match Document::load_mem(buffer) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

pub async fn create_task(
    pool: &Pool,
    files: Vec<&TempFile>,
    task_id: String,
    user_id: String,
    api_key: &String,
) -> Result<TaskResponse, Box<dyn Error>> {
    dotenv().ok();
    let client: Client = pool.get().await?;
    let expiration = env::var("EXPIRATION").ok().map(|val| val).or(None);
    let created_at: DateTime<Utc> = Utc::now();
    let mut file_responses = Vec::new();
    let mut total_size = 0;
    let mut total_pages = 0;

    let bucket_name = env::var("INGEST_SERVER__BUCKET").expect("INGEST_SERVER__BUCKET must be set");

    let mut successful_files = Vec::new();
    for file in &files {
        let file_id = Uuid::new_v4().to_string();

        let buffer: Vec<u8> = std::fs::read(file.file.path())?;

        if is_valid_pdf(&buffer)? {
            let file_size = file.size;

            let page_count = match Document::load_mem(&buffer) {
                Ok(doc) => doc.get_pages().len() as i32,
                Err(_) => 0,
            };
            total_size += file_size;
            total_pages += page_count;
            let file_name = file
                .file_name
                .as_ref()
                .map(String::as_str)
                .unwrap_or("unknown.pdf");
            let s3_path = format!(
                "s3://{}/{}/{}/{}/{}",
                bucket_name, user_id, task_id, file_id, file_name
            );
            let xml_s3_path = s3_path.replace(".pdf", ".xml");

            if upload_to_s3(&s3_path, file_name, buffer, expiration.as_deref()).await? {
                file_responses.push(FileResponse {
                    id: Some(file_id.clone()),
                    status: Some(Status::Starting),
                    message: Some("File is a valid PDF".to_string()),
                    input_url: None,
                    output_url: None,
                });

                client
                    .execute(
                        "INSERT INTO ingestion_files (file_id, task_id, file_name, file_size, page_count, created_at, status, input_location, output_location, expiration_time) 
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) 
                         ON CONFLICT (file_id) DO NOTHING",
                        &[
                            &file_id,
                            &task_id,
                            &file.file_name,
                            &(file_size as i64),
                            &(page_count as i32),
                            &created_at,
                            &Status::Starting,
                            &s3_path,
                            &xml_s3_path,
                            &(expiration.clone().map(|exp| Utc::now() + humantime::parse_duration(&exp).unwrap_or_default())),
                        ],
                    )
                    .await?;

                successful_files.push((file_id, s3_path));
            } else {
                file_responses.push(FileResponse {
                    id: Some(file_id),
                    status: Some(Status::Failed),
                    message: Some("Failed to upload file to S3".to_string()),
                    input_url: None,
                    output_url: None,
                });
            }
        } else {
            file_responses.push(FileResponse {
                id: Some(file_id),
                status: Some(Status::Failed),
                message: Some("File is not a valid PDF".to_string()),
                input_url: None,
                output_url: None,
            });
        }
    }

    let ingest_server_url = env::var("INGEST_SERVER__URL").expect("INGEST_SERVER__URL must be set");
    let response = TaskResponse {
        task_id: task_id.clone(),
        status: Status::starting,
        created_at,
        finished_at: None,
        message: "Ingestion started".to_string(),
        files: file_responses,
        url: Some(format!("{}/task/{}", ingest_server_url, task_id)),
    };

    client
        .execute(
            "INSERT INTO ingestion_tasks (task_id, file_count, total_size, total_pages, created_at, finished_at, api_key, url, status) 
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
             ON CONFLICT (task_id) DO NOTHING",
            &[
                &task_id,
                &(files.len() as i32),
                &(total_size as i64),
                &total_pages,
                &created_at,
                &Option::<DateTime<Utc>>::None,
                &api_key,
                &format!("{}/task/{}", ingest_server_url, task_id),
                &Status::Starting,
            ],
        )
        .await?;

    let xml_queue = env::var("XML__QUEUE").expect("XML__QUEUE must be set");
    let extraction_payloads: Vec<ProducePayload> = successful_files
        .into_iter()
        .map(|(file_id, s3_path)| {
            ProducePayload {
                queue_name: xml_queue.clone(),
                publish_channel: None,
                payload: serde_json::to_value(ExtractionPayload {
                    input_location: s3_path.to_string(),
                    output_location: s3_path.replace(".pdf", ".xml").to_string(),
                    expiration: Some(Duration::from_secs(3600)), // 1 hour in seconds
                    file_id,
                    task_id: task_id.clone(),
                })
                .unwrap(),
                max_attempts: None,
                item_id: Uuid::new_v4().to_string(),
            }
        })
        .collect();

    let result = produce(extraction_payloads).await;

    match result {
        Ok(_) => Ok(response),
        Err(_) => {
            let failed_response = TaskResponse {
                task_id: response.task_id,
                status: Status::Failed,
                created_at: response.created_at,
                finished_at: Some(Utc::now()),
                message: "Failed to send extraction payload".to_string(),
                files: response
                    .files
                    .into_iter()
                    .map(|mut file| {
                        file.status = Some(Status::Failed);
                        file
                    })
                    .collect(),
                url: response.url,
            };
            Ok(failed_response)
        }
    }
}
