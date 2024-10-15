use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::ExtractionPayload;
use crate::models::server::segment::{ PdlaSegment, Segment };
use crate::models::server::task::Status;
use crate::utils::configs::extraction_config::Config;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::services::pdf::split_pdf;
use crate::utils::services::pdla::pdla_extraction;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{ download_to_tempfile, upload_to_s3 };
use crate::utils::workers::{ log::log_task, payload::produce_extraction_payloads };
use chrono::Utc;
use std::io::Write;
use std::path::PathBuf;
use tempdir::TempDir;
use tempfile::NamedTempFile;

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client: aws_sdk_s3::Client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_payload.task_id.clone();
    let pg_pool = create_pool();

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        let input_file = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_payload.input_location,
            None
        ).await?;

        let mut split_temp_files: Vec<PathBuf> = Vec::new();
        let split_temp_dir = TempDir::new("split_pdf")?;

        if let Some(batch_size) = extraction_payload.batch_size {
            split_temp_files = split_pdf(
                &input_file.path(),
                batch_size as usize,
                split_temp_dir.path()
            )?;
        } else {
            split_temp_files.push(input_file.path().to_path_buf());
        }

        let mut combined_output: Vec<Segment> = Vec::new();
        let mut page_offset: u32 = 0;
        let mut batch_number: i32 = 0;

        for temp_file in &split_temp_files {
            batch_number += 1;
            let segmentation_message = if split_temp_files.len() > 1 {
                format!("Segmenting | Batch {} of {}", batch_number, split_temp_files.len())
            } else {
                "Segmenting".to_string()
            };

            log_task(
                task_id.clone(),
                Status::Processing,
                Some(segmentation_message),
                None,
                &pg_pool
            ).await?;

            let temp_file_path = temp_file.to_path_buf();

            let pdla_response = pdla_extraction(
                &temp_file_path,
                extraction_payload.model.clone()
            ).await?;
            let pdla_segments: Vec<PdlaSegment> = serde_json::from_str(&pdla_response)?;
            let mut segments: Vec<Segment> = pdla_segments
                .iter()
                .map(|pdla_segment| pdla_segment.to_segment())
                .collect();

            for item in &mut segments {
                item.page_number += page_offset;
            }
            combined_output.extend(segments);
            page_offset += extraction_payload.batch_size.unwrap_or(1) as u32;
        }

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&combined_output)?.as_bytes())?;

        upload_to_s3(
            &s3_client,
            &extraction_payload.output_location,
            &output_temp_file.path()
        ).await?;

        if output_temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(output_temp_file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
            }
        }

        Ok(())
    }).await;

    match result {
        Ok(_) => {
            println!("Task succeeded");
            let config = Config::from_env()?;
            produce_extraction_payloads(config.queue_ocr, extraction_payload).await?;
            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing task: {:?}", e);
            if payload.attempt >= payload.max_attempts {
                eprintln!("Task failed after {} attempts", payload.max_attempts);
                log_task(
                    task_id.clone(),
                    Status::Failed,
                    Some("Segmentation failed".to_string()),
                    Some(Utc::now()),
                    &pg_pool
                ).await?;
            }
            Err(e)
        }
    }
}
