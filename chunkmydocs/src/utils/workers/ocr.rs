use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::ExtractionPayload;
use crate::models::server::segment::Segment;
use crate::models::server::task::Status;
use crate::utils::db::deadpool_postgres::{create_pool, Client, Pool};
use crate::utils::json2mkd::json_2_mkd::hierarchical_chunking;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::{DateTime, Utc};
use std::io::{Read, Write};
use tempfile::NamedTempFile;

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_item: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_item.task_id.clone();
    let pg_pool = create_pool();


    let result: Result<(), Box<dyn std::error::Error>> = (async {
        let pdf_file: NamedTempFile = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_item.input_location,
            None,
        )
        .await?;

        let output_file: NamedTempFile = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_item.output_location,
            None,
        )
        .await?;
        
        let pdf_file_path = pdf_file.path().to_path_buf();

        let mut file_contents = String::new();
        output_file.as_file().read_to_string(&mut file_contents)?;

        let incomplete_segments: Vec<Segment> = match serde_json::from_str(&file_contents) {
            Ok(segments) => segments,
            Err(e) => {
                return Err(Box::new(e) as Box<dyn std::error::Error>);
            }
        };

        let processing_message = format!(
            "Processing | OCR: {}",
            extraction_item.configuration.ocr_strategy
        );

        log_task(
            task_id.clone(),
            Status::Processing,
            Some(processing_message),
            None,
            &pg_pool,
        )
        .await?;

        let segments: Vec<Segment> =
            process_segments(&pdf_file_path, &incomplete_segments, &extraction_item).await?;

        let chunks = hierarchical_chunking(segments, extraction_item.target_chunk_length).await?;

        let mut finalized_file = NamedTempFile::new()?;
        finalized_file.write_all(serde_json::to_string(&chunks)?.as_bytes())?;

        upload_to_s3(
            &s3_client,
            &extraction_item.output_location,
            &finalized_file.path(),
        )
        .await?;

        if pdf_file.path().exists() {
            if let Err(e) = std::fs::remove_file(pdf_file.path()) {
                eprintln!("Error deleting temporary PDF file: {:?}", e);
            }
        }
        if output_file.path().exists() {
            if let Err(e) = std::fs::remove_file(output_file.path()) {
                eprintln!("Error deleting temporary output file: {:?}", e);
            }
        }
        if finalized_file.path().exists() {
            if let Err(e) = std::fs::remove_file(finalized_file.path()) {
                eprintln!("Error deleting temporary finalized file: {:?}", e);
            }
        }

        Ok(())
    })
    .await;

    match result {
        Ok(_) => {
            println!("Task succeeded");
            log_task(
                task_id.clone(),
                Status::Succeeded,
                Some("Task succeeded".to_string()),
                Some(Utc::now()),
                &pg_pool,
            )
            .await?;
            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing task: {:?}", e);
            if payload.attempt >= payload.max_attempts {
                eprintln!("Task failed after {} attempts", payload.max_attempts);
                log_task(
                    task_id.clone(),
                    Status::Failed,
                    Some("OCR failed".to_string()),
                    Some(Utc::now()),
                    &pg_pool,
                )
                .await?;
            }
            Err(e)
        }
    }
}
