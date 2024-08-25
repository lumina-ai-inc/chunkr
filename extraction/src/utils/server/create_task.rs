use crate::models::rrq::produce::ProducePayload;
use crate::models::{
    extraction::extract::{ExtractionPayload, ModelInternal},
    extraction::task::{Status, TaskResponse},
};
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::rrq::service::produce;
use crate::utils::storage_service::services::upload_to_s3;
use actix_multipart::form::tempfile::TempFile;
use chrono::{DateTime, Utc};
use dotenvy::dotenv;
use humantime;
use lopdf::Document;
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

async fn produce_extraction_payloads(
    extraction_payload: ExtractionPayload,
) -> Result<(), Box<dyn Error>> {
    let document_extraction_queue =
        env::var("EXTRACTION__QUEUE").expect("EXTRACTION__QUEUE must be set");

    let produce_payload = ProducePayload {
        queue_name: document_extraction_queue.clone(),
        publish_channel: None,
        payload: serde_json::to_value(extraction_payload).unwrap(),
        max_attempts: None,
        item_id: Uuid::new_v4().to_string(),
    };

    produce(vec![produce_payload]).await?;

    Ok(())
}

pub async fn create_task(
    pool: &Pool,
    file: &TempFile,
    task_id: String,
    user_id: String,
    api_key: &String,
    model: ModelInternal,
) -> Result<TaskResponse, Box<dyn Error>> {
    dotenv().ok();
    let mut client: Client = pool.get().await?;
    let expiration = env::var("INGEST_SERVER__EXPIRATION").ok().or(None);
    let created_at: DateTime<Utc> = Utc::now();
    let expiration_time: Option<DateTime<Utc>> = expiration.clone().map(|exp| {
        let std_duration: Duration = humantime::parse_duration(&exp).unwrap();
        Utc::now() + std_duration
    });

    let bucket_name = env::var("INGEST_SERVER__BUCKET").expect("INGEST_SERVER__BUCKET must be set");
    let ingest_batch_size = env::var("INGEST_BATCH_SIZE").expect("INGEST_BATCH_SIZE must be set");
    let ingest_server_url = env::var("INGEST_SERVER__URL").expect("INGEST_SERVER__URL must be set");
    let task_url = format!("{}/task/{}", ingest_server_url, task_id);

    let file_id = Uuid::new_v4().to_string();
    let buffer: Vec<u8> = std::fs::read(file.file.path())?;

    if is_valid_pdf(&buffer)? {
        let file_size = file.size;
        let page_count = match Document::load_mem(&buffer) {
            Ok(doc) => doc.get_pages().len() as i32,
            Err(_) => {
                return Err("Unable to count pages".into());
            }
        };

        let file_name = file.file_name.as_deref().unwrap_or("unknown.pdf");
        let s3_path = format!(
            "s3://{}/{}/{}/{}/{}",
            bucket_name, user_id, task_id, file_id, file_name
        );
        let output_extension = model.get_extension();
        let output_s3_path = s3_path.replace(".pdf", &format!(".{}", output_extension));

        if upload_to_s3(&s3_path, file_name, buffer, None).await? {
            let tx = client.transaction().await?;

            tx.execute(
                "INSERT INTO ingestion_tasks (task_id, file_count, total_size, total_pages, created_at, finished_at, api_key, url, status, model, expiration_time) 
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) 
                 ON CONFLICT (task_id) DO NOTHING",
                &[
                    &task_id,
                    &1i32,
                    &(file_size as i64),
                    &page_count,
                    &created_at,
                    &None::<String>,
                    &api_key,
                    &format!("{}/task/{}", ingest_server_url, task_id),
                    &Status::Starting.to_string(),
                    &model.to_string(),
                    &expiration_time
                ]
            ).await?;

            tx.execute(
                "INSERT INTO ingestion_files (file_id, task_id, file_name, file_size, page_count, created_at, status, input_location, output_location, model) 
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) 
                 ON CONFLICT (file_id) DO NOTHING",
                &[
                    &file_id,
                    &task_id,
                    &file.file_name,
                    &(file_size as i64),
                    &page_count,
                    &created_at,
                    &Status::Starting.to_string(),
                    &s3_path,
                    &output_s3_path,
                    &model.to_string(),
                ]
            ).await?;

            tx.commit().await?;

            let extraction_payload = ExtractionPayload {
                model: model.clone(),
                input_location: s3_path,
                output_location: output_s3_path,
                expiration: None,
                batch_size: Some(ingest_batch_size.parse::<i32>().unwrap()),
                file_id,
                task_id: task_id.clone(),
            };

            produce_extraction_payloads(extraction_payload).await?;

            Ok(TaskResponse {
                task_id: task_id.clone(),
                status: Status::Starting,
                created_at,
                finished_at: None,
                expiration_time,
                file_url: None,
                task_url: Some(task_url),
                message: "Extraction started".to_string(),
                model: model.to_external(),
            })
        } else {
            Err("Failed to upload file".into())
        }
    } else {
        Err("Not a valid PDF".into())
    }
}
