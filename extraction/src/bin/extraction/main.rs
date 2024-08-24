use chrono::Utc;
use extraction::models::extraction::extraction::ExtractionPayload;
use extraction::models::extraction::extraction::ModelInternal;
use extraction::models::extraction::task::Status;
use extraction::models::rrq::{produce::ProducePayload, queue::QueuePayload};
use extraction::utils::rrq::{consumer::consumer, service::produce};
use extraction::utils::storage_service::services::{download_to_tempfile, upload_to_s3};
use humantime::format_duration;
use serde_json::json;
use std::{fs, path::PathBuf};
use uuid::Uuid;

mod extraction_config;
mod grobid;
mod pdf;
mod pdla;
use crate::grobid::grobid_extraction;
use crate::pdla::pdla_extraction;
use extraction_config::Config;

pub async fn log_task(
    task_id: String,
    file_id: String,
    status: Status,
    message: Option<String>,
    finished_at: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;

    println!("Prepared status: {:?}", status);
    println!("Prepared task_id: {}", task_id);
    println!("Prepared file_id: {}", file_id);

    let task_query = format!(
        "UPDATE ingestion_tasks SET status = '{:?}', message = '{}', finished_at = '{:?}' WHERE task_id = '{}'",
        status,
        message.unwrap_or_default(),
        finished_at.unwrap_or_default(),
        task_id
    );

    let files_query = format!(
        "UPDATE ingestion_files SET status = '{:?}' WHERE task_id = '{}' AND file_id = '{}'",
        status, task_id, file_id
    );

    let payloads = vec![
        ProducePayload {
            queue_name: config.extraction_queue.clone(),
            publish_channel: None,
            payload: json!(task_query),
            max_attempts: Some(3),
            item_id: Uuid::new_v4().to_string(),
        },
        ProducePayload {
            queue_name: config.extraction_queue.clone(),
            publish_channel: None,
            payload: json!(files_query),
            max_attempts: Some(3),
            item_id: Uuid::new_v4().to_string(),
        },
    ];

    produce(payloads).await?;

    Ok(())
}

async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    let extraction_item: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_item.task_id.clone();
    let file_id = extraction_item.file_id.clone();

    println!("{:?}", extraction_item.clone());

    log_task(
        task_id.clone(),
        file_id.clone(),
        Status::Processing,
        Some(format!(
            "Task processing | Retry ({}/{})",
            payload.attempt, payload.max_attempts
        )),
        None,
    )
    .await?;

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        let temp_file = download_to_tempfile(&extraction_item.input_location).await?;
        println!("Downloaded file to {:?}", temp_file.path());

        let output_path: PathBuf;

        if extraction_item.model == ModelInternal::Grobid {
            output_path = grobid_extraction(temp_file.path()).await?;
        } else if extraction_item.model == ModelInternal::Pdla
            || extraction_item.model == ModelInternal::PdlaFast
        {
            output_path = pdla_extraction(
                temp_file.path(),
                extraction_item.model,
                extraction_item.batch_size,
            )
            .await?;
        } else {
            return Err("Invalid model".into());
        }

        upload_to_s3(
            &extraction_item.output_location,
            output_path.clone().to_str().unwrap(),
            fs::read(output_path)?,
            extraction_item
                .expiration
                .map(|d| format_duration(d).to_string())
                .as_deref(),
        )
        .await?;

        if temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(temp_file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
            }
        }

        Ok(())
    })
    .await;

    match result {
        Ok(_) => {
            log_task(
                task_id.clone(),
                file_id.clone(),
                Status::Succeeded,
                Some("Task succeeded".to_string()),
                Some(Utc::now().to_string()),
            )
            .await?;
            println!("Task succeeded");
            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing task: {:?}", e);
            if payload.attempt >= payload.max_attempts {
                log_task(
                    task_id.clone(),
                    file_id.clone(),
                    Status::Failed,
                    Some(e.to_string()),
                    Some(Utc::now().to_string()),
                )
                .await?;
                println!("Task failed");
            }
            Err(e)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    consumer(process, config.extraction_queue, 1, 600).await?;
    Ok(())
}
