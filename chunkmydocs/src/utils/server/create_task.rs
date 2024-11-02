use crate::models::auth::auth::UserInfo;
use crate::models::{
    server::extract::{Configuration, ExtractionPayload},
    server::task::{Status, TaskResponse},
};
use crate::utils::configs::extraction_config::Config;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::services::payload::produce_extraction_payloads;
use crate::utils::storage::services::{generate_presigned_url, upload_to_s3};
use actix_multipart::form::tempfile::TempFile;
use aws_sdk_s3::Client as S3Client;
use chrono::{DateTime, Utc};
use std::{
    error::Error,
    path::{Path, PathBuf},
    time::Instant,
};
use uuid::Uuid;

pub async fn create_task(
    pool: &Pool,
    s3_client: &S3Client,
    file: &TempFile,
    user_info: &UserInfo,
    configuration: &Configuration,
) -> Result<TaskResponse, Box<dyn Error>> {
    let start_time = Instant::now();

    let task_id = Uuid::new_v4().to_string();
    let user_id = user_info.user_id.clone();
    let api_key = user_info.api_key.clone();
    let model = configuration.model.clone();
    let model_internal = model.to_internal();
    let target_chunk_length = configuration.target_chunk_length.unwrap_or(512);
    let created_at = Utc::now();
    let client: Client = pool.get().await?;
    let config = Config::from_env()?;
    let expiration = config.task_expiration;
    let expiration_time: Option<DateTime<Utc>> = expiration.map(|exp| Utc::now() + exp);
    let bucket_name = config.s3_bucket;
    let ingest_batch_size = config.batch_size;
    let base_url = config.base_url;
    let task_url = format!("{}/api/v1/task/{}", base_url, task_id);

    let final_output_path: PathBuf = PathBuf::from(file.file.path());

    let page_count: i32 = 0;
    let file_size = file.size;

    let file_name = file.file_name.as_deref().unwrap_or("unknown.pdf");

    let path = Path::new(file_name);
    let file_stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("pdf");

    let input_location = format!("s3://{}/{}/{}/{}", bucket_name, user_id, task_id, file_name);
    let pdf_location = format!(
        "s3://{}/{}/{}/{}.pdf",
        bucket_name, user_id, task_id, file_stem
    );

    let output_extension = model_internal.get_extension();
    let output_location = format!(
        "s3://{}/{}/{}/{}.{}",
        bucket_name, user_id, task_id, file_stem, output_extension
    );

    let image_folder_location =
        format!("s3://{}/{}/{}/{}", bucket_name, user_id, task_id, "images");

    let message = "Task queued".to_string();

    match upload_to_s3(s3_client, &input_location, &final_output_path).await {
        Ok(_) => {
            let configuration_json = serde_json::to_string(configuration)?;

            match client
                .execute(
                    "INSERT INTO TASKS (
                    task_id, user_id, api_key, file_name, file_size, 
                    page_count, segment_count, expires_at,
                    status, task_url, input_location, output_location, image_folder_location,
                    configuration, message, pdf_location, input_file_type
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
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
                        &image_folder_location,
                        &configuration_json,
                        &message,
                        &pdf_location,
                        &extension,
                    ],
                )
                .await
            {
                Ok(_) => {
                    println!(
                        "Time taken to insert into database: {:?}",
                        start_time.elapsed().as_secs_f32()
                    );
                }
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

            let extraction_payload = ExtractionPayload {
                user_id: user_id.clone(),
                model: model_internal,
                input_location: input_location.clone(),
                pdf_location: pdf_location.clone(),
                output_location,
                image_folder_location,
                expiration: None,
                batch_size: Some(ingest_batch_size),
                task_id: task_id.clone(),
                target_chunk_length: Some(target_chunk_length),
                configuration: configuration.clone(),
                file_name: file_name.to_string(),
                page_count: Some(page_count),
            };

            match produce_extraction_payloads(config.queue_preprocess, extraction_payload).await {
                Ok(_) => {
                    println!(
                        "Time taken to produce extraction payloads: {:?}",
                        start_time.elapsed().as_secs_f32()
                    );
                }
                Err(e) => {
                    match client
                        .execute("DELETE FROM TASKS WHERE task_id = $1", &[&task_id])
                        .await
                    {
                        Ok(_) => {
                            println!(
                                "Time taken to delete task from database: {:?}",
                                start_time.elapsed().as_secs_f32()
                            );
                        }
                        Err(e) => {
                            println!("Error deleting task from database: {:?}", e);
                        }
                    }
                    return Err(e);
                }
            }

            println!(
                "Time taken to produce extraction payloads: {:?}",
                start_time.elapsed().as_secs_f32()
            );

            let input_file_url =
                match generate_presigned_url(s3_client, &input_location, None).await {
                    Ok(response) => Some(response),
                    Err(_e) => {
                        return Err("Error getting input file url".into());
                    }
                };

            println!(
                "Time taken to generate presigned url: {:?}",
                start_time.elapsed().as_secs_f32()
            );

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
                file_name: Some(file_name.to_string()),
                page_count: Some(page_count),
                pdf_url: None,
            })
        }
        Err(e) => Err(e),
    }
}
