use crate::models::auth::auth::UserInfo;
use crate::models::rrq::produce::ProducePayload;
use crate::models::{
    server::extract::{Configuration, ExtractionPayload, SegmentationModel},
    server::task::{Status, TaskResponse},
};
use crate::task::pdf::convert_to_pdf;
use crate::utils::configs::extraction_config::Config;
use crate::utils::db::deadpool_postgres::{Client, Pool};
use crate::utils::rrq::service::produce;
use crate::utils::storage::services::{generate_presigned_url, upload_to_s3};
use actix_multipart::form::tempfile::TempFile;
use aws_sdk_s3::Client as S3Client;
use chrono::{DateTime, Utc};
use lopdf::Document;
use mime_guess::MimeGuess;
use std::error::Error;
use std::path::Path;
use std::path::PathBuf;
use uuid::Uuid;

fn detect_file_type(file_path: &Path) -> Result<String, Box<dyn Error>> {
    let guess = MimeGuess::from_path(file_path);
    let mime_type = match guess.first() {
        Some(mime) => mime.to_string(),
        None => "application/octet-stream".to_string(),
    };
    Ok(mime_type)
}

fn is_valid_file_type(
    file_path: &Path,
    original_file_name: &str,
) -> Result<(bool, String), Box<dyn Error>> {
    let extension = Path::new(original_file_name)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    println!("Extension: {}", extension);

    // Simplify file type validation based on extension
    let is_valid = match extension.to_lowercase().as_str() {
        "pdf" | "docx" | "doc" | "pptx" | "ppt" => true,
        _ => false,
    };
    
    // We don't need to create a temporary file or detect MIME type
    // since we're only checking the extension
    
    Ok((is_valid, format!("application/{}", extension)))
}

async fn produce_extraction_payloads(
    extraction_payload: ExtractionPayload,
) -> Result<(), Box<dyn Error>> {
    let config = Config::from_env()?;

    let queue_name = match extraction_payload.model {
        SegmentationModel::PdlaFast => config.extraction_queue_fast,
        SegmentationModel::Pdla => config.extraction_queue_high_quality,
    };

    let queue_name = queue_name.ok_or_else(|| "Queue name not configured".to_string())?;

    let produce_payload = ProducePayload {
        queue_name,
        publish_channel: None,
        payload: serde_json::to_value(extraction_payload).unwrap(),
        max_attempts: None,
        item_id: Uuid::new_v4().to_string(),
    };

    produce(vec![produce_payload]).await?;

    Ok(())
}
use tempfile::NamedTempFile;
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
    let created_at = Utc::now();
    let client: Client = pool.get().await?;
    let config = Config::from_env()?;
    let expiration = config.task_expiration;
    let expiration_time: Option<DateTime<Utc>> = expiration.map(|exp| Utc::now() + exp);
    let bucket_name = config.s3_bucket;
    let ingest_batch_size = config.batch_size;
    let base_url = config.base_url;
    let task_url = format!("{}/api/v1/task/{}", base_url, task_id);

    let file_name = file.file_name.as_deref().unwrap_or("unknown");
    let original_path = PathBuf::from(file.file.path());
    let (is_valid, detected_mime_type) = is_valid_file_type(&file.file.path(), file_name)?;
    if !is_valid {
        return Err(format!("Not a valid file type: {}", detected_mime_type).into());
    }

    let extension = file
        .file_name
        .as_deref()
        .unwrap_or("")
        .split('.')
        .last()
        .unwrap_or("tmp");


    let mut final_output_path: PathBuf = original_path.clone();
    println!("Initial final output path: {:?}", final_output_path);
    let final_output_file = tempfile::NamedTempFile::new().unwrap();
    let output_file = NamedTempFile::new().unwrap();
    let output_path = output_file.path().to_path_buf();

    if extension != "pdf" {
        let new_path = original_path.with_extension(extension).clone();
        println!("New path: {:?}", new_path);
    
        std::fs::rename(&original_path, &new_path)?;
        println!("File renamed from {:?} to {:?}", original_path, new_path);
    
        let input_path = new_path;
        println!("Input path: {:?}", input_path);
    
        println!("Output path: {:?}", output_path);
    
        println!("Converting file to PDF");
        let result = convert_to_pdf(&input_path, &output_path).await;
        final_output_path = final_output_file.path().to_path_buf();
        println!("Final output path after conversion: {:?}", final_output_path);

        match result {
            Ok(_) => {
                std::fs::copy(&output_path, &final_output_path).unwrap();
                println!("PDF conversion successful. Output saved to: {:?}", final_output_path);
            }
            Err(e) => {
                println!("PDF conversion failed: {:?}", e);
                panic!("PDF conversion failed: {:?}", e);
            }
        }
    } else {
        println!("File is already in PDF format. No conversion needed.");
    }

    println!("Final output path: {:?}", final_output_path);
    let page_count = match Document::load(&final_output_path) {
        Ok(doc) => doc.get_pages().len() as i32,
        Err(e) => return Err(format!("Failed to get page count: {}", e).into()),
    };
    let file_size = file.size;
    println!("File size: {}", file_size);

    let file_name = file
        .file_name
        .as_deref()
        .unwrap_or("unknown.pdf")
        .to_string();
    println!("Original file name: {}", file_name);

    let file_name = if file_name.ends_with(".pdf") {
        file_name
    } else {
        format!(
            "{}.pdf",
            file_name.trim_end_matches(|c| c == '.' || char::is_alphanumeric(c))
        )
    };
    println!("Processed file name: {}", file_name);

    let input_location = format!("s3://{}/{}/{}/{}", bucket_name, user_id, task_id, file_name);
    println!("Input location: {}", input_location);

    let output_extension = model_internal.get_extension();
    println!("Output extension: {}", output_extension);

    let output_location = input_location.replace(".pdf", &format!(".{}", output_extension));
    println!("Output location: {}", output_location);

    let image_folder_location =
        format!("s3://{}/{}/{}/{}", bucket_name, user_id, task_id, "images");
    println!("Image folder location: {}", image_folder_location);

    let message = "Task queued".to_string();
    println!("Task status message: {}", message);

    match upload_to_s3(s3_client, &input_location, &final_output_path).await {
        Ok(_) => {
            println!("Converting configuration to JSON");
            let configuration_json = serde_json::to_string(configuration)?;

            println!("Inserting task into database");
            match client
                .execute(
                    "INSERT INTO TASKS (
                    task_id, user_id, api_key, file_name, file_size, 
                    page_count, segment_count, expires_at,
                    status, task_url, input_location, output_location, image_folder_location,
                    configuration, message
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
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
                    ],
                )
                .await
            {
                Ok(_) => {
                    println!("Task inserted successfully");
                }
                Err(e) => {
                    println!("Error inserting task: {}", e);
                    if e.to_string().contains("usage limit exceeded") {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "429 Rate Limit Error: Usage limit exceeded",
                        )));
                    } else {
                        return Err(Box::new(e) as Box<dyn Error>);
                    }
                }
            };

            println!("Creating extraction payload");
            let extraction_payload = ExtractionPayload {
                user_id: user_id.clone(),
                model: model_internal,
                input_location: input_location.clone(),
                output_location,
                image_folder_location,
                expiration: None,
                batch_size: Some(ingest_batch_size),
                task_id: task_id.clone(),
                target_chunk_length: Some(target_chunk_length),
                configuration: configuration.clone(),
            };

            println!("Producing extraction payloads");
            produce_extraction_payloads(extraction_payload).await?;

            println!("Generating presigned URL for input file");
            let input_file_url =
                match generate_presigned_url(s3_client, &input_location, None).await {
                    Ok(response) => {
                        println!("Presigned URL generated successfully");
                        Some(response)
                    },
                    Err(e) => {
                        println!("Error getting input file url: {}", e);
                        return Err("Error getting input file url".into());
                    }
                };

            println!("Creating TaskResponse");
            Ok(TaskResponse {
                task_id: task_id.clone(),
                status: Status::Starting,
                created_at: created_at,
                finished_at: None,
                expires_at: expiration_time,
                output: None,
                input_file_url,
                task_url: Some(task_url),
                message,
                configuration: configuration.clone(),
                file_name: Some(file_name.to_string()),
                page_count: Some(page_count),
            })
        }
        Err(e) => Err(e),
    }
}
