use crate::models::rrq::produce::ProducePayload;
use crate::models::{
    extraction::extract::{ExtractionPayload, ModelInternal},
    extraction::task::{Status, TaskResponse},
};
use crate::utils::configs::extraction_config::Config;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::rrq::service::produce;
use crate::utils::storage::services::{generate_presigned_url, upload_to_s3};
use actix_multipart::form::tempfile::TempFile;
use aws_sdk_s3::Client as S3Client;
use chrono::{DateTime, Utc};
use lopdf::Document;
use std::error::Error;
use uuid::Uuid;

async fn validate_usage(
    client: &Client,
    api_key: &String,
    page_count: i32,
) -> Result<(), Box<dyn Error>> {
    // Check current usage and limit
    let usage_row = client.query_one(
        "SELECT COALESCE(SUM(usage), 0) as total_usage FROM public.api_key_usage WHERE api_key = $1 AND usage_type = 'FREE' AND service = 'EXTRACTION'",
        &[&api_key]
    ).await?;
    let current_usage: i64 = usage_row.get("total_usage");

    let limit_row = client.query_opt(
        "SELECT usage_limit FROM public.api_key_limit WHERE api_key = $1 AND usage_type = 'FREE' AND service = 'EXTRACTION' LIMIT 1",
        &[&api_key]
    ).await?;
    let usage_limit: i32 = limit_row.map(|row| row.get("usage_limit")).unwrap_or(0);

    if current_usage + i64::from(page_count) > i64::from(usage_limit) {
        let exceeded_by = (current_usage + i64::from(page_count)) - i64::from(usage_limit);
        return Err(Box::new(actix_web::error::ErrorTooManyRequests(format!(
            "Adding a task with {} pages would exceed the usage limit of {} pages by {} pages.",
            page_count, usage_limit, exceeded_by
        ))));
    }

    Ok(())
}

fn is_valid_pdf(buffer: &[u8]) -> Result<bool, lopdf::Error> {
    match Document::load_mem(buffer) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

async fn produce_extraction_payloads(
    extraction_payload: ExtractionPayload,
) -> Result<(), Box<dyn Error>> {
    let config = Config::from_env()?;

    let produce_payload = ProducePayload {
        queue_name: config.extraction_queue,
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
    s3_client: &S3Client,
    file: &TempFile,
    task_id: String,
    user_id: String,
    api_key: &String,
    model: ModelInternal,
) -> Result<TaskResponse, Box<dyn Error>> {
    let mut client: Client = pool.get().await?;
    let config = Config::from_env()?;
    let expiration = config.task_expiration;
    let created_at: DateTime<Utc> = Utc::now();
    let expiration_time: Option<DateTime<Utc>> = expiration.map(|exp| Utc::now() + exp);

    let bucket_name = config.s3_bucket;
    let ingest_batch_size = config.batch_size;
    let base_url = config.base_url;
    let task_url = format!("{}/task/{}", base_url, task_id);

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

        validate_usage(&client, &api_key, page_count).await?;

        let file_name = file.file_name.as_deref().unwrap_or("unknown.pdf");
        let input_location = format!(
            "s3://{}/{}/{}/{}/{}",
            bucket_name, user_id, task_id, file_id, file_name
        );
        let output_extension = model.get_extension();
        let output_location = input_location.replace(".pdf", &format!(".{}", output_extension));

        match upload_to_s3(s3_client, &input_location, file.file.path()).await {
            Ok(_) => {
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
                        &task_url,
                        &Status::Starting.to_string(),
                        &model.to_string(),
                        &expiration_time,
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
                        &input_location,
                        &output_location,
                        &model.to_string(),
                    ]
                ).await?;

                tx.commit().await?;

                let extraction_payload = ExtractionPayload {
                    model: model.clone(),
                    input_location: input_location.clone(),
                    output_location,
                    expiration: None,
                    batch_size: Some(ingest_batch_size),
                    file_id,
                    task_id: task_id.clone(),
                };

                produce_extraction_payloads(extraction_payload).await?;

                let input_file_url =
                    match generate_presigned_url(s3_client, &input_location, None).await {
                        Ok(response) => Some(response),
                        Err(e) => {
                            println!("Error getting input file url: {}", e);
                            return Err("Error getting input file url".into());
                        }
                    };

                Ok(TaskResponse {
                    task_id: task_id.clone(),
                    status: Status::Starting,
                    created_at,
                    finished_at: None,
                    expiration_time,
                    output_file_url: None,
                    input_file_url,
                    task_url: Some(task_url),
                    message: "Task queued".to_string(),
                    model: model.to_external(),
                })
            }
            Err(e) => Err(e),
        }
    } else {
        Err("Not a valid PDF".into())
    }
}
