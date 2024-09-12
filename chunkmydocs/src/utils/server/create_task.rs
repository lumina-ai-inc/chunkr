use crate::models::auth::auth::UserInfo;
use crate::models::rrq::produce::ProducePayload;
use crate::models::{
    server::extract::{Configuration, ExtractionPayload},
    server::task::{Status, TaskResponse},
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
    user_info: &UserInfo,
    configuration: &Configuration,
) -> Result<TaskResponse, Box<dyn Error>> {
    let task_id = Uuid::new_v4().to_string();

    let user_id = user_info.user_id.clone();
    let api_key = user_info.api_key.clone();
    let model = configuration.model.clone();
    let model_internal = model.to_internal();
    let target_chunk_length = configuration.target_chunk_length.unwrap_or(512);

    let client: Client = pool.get().await?;
    let config = Config::from_env()?;
    let expiration = config.task_expiration;
    let created_at: DateTime<Utc> = Utc::now();
    let expiration_time: Option<DateTime<Utc>> = expiration.map(|exp| Utc::now() + exp);
    let bucket_name = config.s3_bucket;
    let ingest_batch_size = config.batch_size;
    let base_url = config.base_url;
    let task_url = format!("{}/task/{}", base_url, task_id);

    let buffer: Vec<u8> = std::fs::read(file.file.path())?;

    if !is_valid_pdf(&buffer)? {
        return Err("Not a valid PDF".into());
    }

    let file_size = file.size;
    let page_count = match Document::load_mem(&buffer) {
        Ok(doc) => doc.get_pages().len() as i32,
        Err(_) => {
            return Err("Unable to count pages".into());
        }
    };

    let file_name = file.file_name.as_deref().unwrap_or("unknown.pdf");
    let input_location = format!("s3://{}/{}/{}/{}", bucket_name, user_id, task_id, file_name);
    let output_extension = model_internal.get_extension();
    let output_location = input_location.replace(".pdf", &format!(".{}", output_extension));

    let message = "Task queued".to_string();

    match upload_to_s3(s3_client, &input_location, file.file.path()).await {
        Ok(_) => {
            let configuration_json = serde_json::to_string(configuration)?;
            client
                .execute(
                    "INSERT INTO TASKS (
                    task_id, user_id, api_key, file_name, file_size, 
                    page_count, segment_count, expires_at,
                    status, task_url, input_location, output_location, 
                    configuration, message
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                ) ON CONFLICT (task_id) DO NOTHING",
                    &[
                        &task_id,
                        &user_id,
                        &api_key,
                        &file_name,
                        &(file_size as i64),
                        &page_count,
                        &0i32,
                        &expiration_time,
                        &Status::Starting.to_string(),
                        &task_url,
                        &input_location,
                        &output_location,
                        &configuration_json,
                        &message,
                    ],
                )
                .await?;

            let extraction_payload = ExtractionPayload {
                model: model_internal,
                input_location: input_location.clone(),
                output_location,
                expiration: None,
                batch_size: Some(ingest_batch_size),
                task_id: task_id.clone(),
                target_chunk_length: Some(target_chunk_length),
                table_ocr: configuration.table_ocr,
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
                expires_at: expiration_time,
                output: None,
                input_file_url,
                task_url: Some(task_url),
                message,
                configuration: configuration.clone(),
            })
        }
        Err(e) => Err(e),
    }
}
