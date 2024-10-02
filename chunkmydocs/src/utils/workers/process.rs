use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::ExtractionPayload;
use crate::models::server::segment::{BaseSegment, PdlaSegment, Segment};
use crate::models::server::task::Status;
use crate::task::pdf::split_pdf;
use crate::task::pdla::pdla_extraction;
use crate::task::process::process_segments;
use crate::utils::db::deadpool_postgres::{create_pool, Client, Pool};
use crate::utils::json2mkd::json_2_mkd::hierarchical_chunking;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::{DateTime, Utc};
use std::{io::Write, path::PathBuf};
use tempdir::TempDir;
use tempfile::NamedTempFile;

pub async fn log_task(
    task_id: String,
    status: Status,
    message: Option<String>,
    finished_at: Option<DateTime<Utc>>,
    pool: &Pool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

    let task_query = format!(
        "UPDATE tasks SET status = '{:?}', message = '{}', finished_at = '{:?}' WHERE task_id = '{}'",
        status,
        message.unwrap_or_default(),
        finished_at.unwrap_or_default(),
        task_id
    );

    client.execute(&task_query, &[]).await?;

    Ok(())
}

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_item: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_item.task_id.clone();

    let pg_pool = create_pool();

    log_task(
        task_id.clone(),
        Status::Processing,
        Some(format!(
            "Task processing | Tries ({}/{})",
            payload.attempt, payload.max_attempts
        )),
        None,
        &pg_pool,
    )
    .await?;

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        let temp_file = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_item.input_location,
            None,
        )
        .await?;

        let mut split_temp_files: Vec<PathBuf> = vec![];
        let split_temp_dir = TempDir::new("split_pdf")?;

        if let Some(batch_size) = extraction_item.batch_size {
            split_temp_files =
                split_pdf(temp_file.path(), batch_size as usize, split_temp_dir.path()).await?;
        } else {
            split_temp_files.push(temp_file.path().to_path_buf());
        }

        let mut combined_output: Vec<Segment> = Vec::new();
        let mut page_offset: u32 = 0;
        let mut batch_number: i32 = 0;

        for temp_file in &split_temp_files {
            batch_number += 1;
            let mut segmentation_message = "Segmenting".to_string();
            let mut processing_message = format!(
                "Processing | OCR: {}",
                extraction_item.configuration.ocr_strategy
            );
            if split_temp_files.len() > 1 {
                segmentation_message = format!(
                    "Segmenting | Batch {} of {}",
                    batch_number,
                    split_temp_files.len()
                );
                processing_message = format!(
                    "Processing | OCR: {} | Batch {} of {}",
                    extraction_item.configuration.ocr_strategy,
                    batch_number,
                    split_temp_files.len()
                );
            }
            log_task(
                task_id.clone(),
                Status::Processing,
                Some(segmentation_message),
                None,
                &pg_pool,
            )
            .await?;

            let temp_file_path = temp_file.as_path().to_path_buf();

            let pdla_response =
                pdla_extraction(&temp_file_path, extraction_item.model.clone()).await?;
            let pdla_segments: Vec<PdlaSegment> = serde_json::from_str(&pdla_response)?;
            let base_segments: Vec<BaseSegment> = pdla_segments
                .iter()
                .map(|pdla_segment| pdla_segment.to_base_segment())
                .collect();
            log_task(
                task_id.clone(),
                Status::Processing,
                Some(processing_message),
                None,
                &pg_pool,
            )
            .await?;

            let mut segments: Vec<Segment> = process_segments(
                &temp_file_path,
                &base_segments,
                &extraction_item
            )
            .await?;

            for item in &mut segments {
                item.page_number = item.page_number + page_offset;
            }
            combined_output.extend(segments);
            page_offset += extraction_item.batch_size.unwrap_or(1) as u32;
        }

        let chunks =
            hierarchical_chunking(combined_output, extraction_item.target_chunk_length).await?;

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&chunks)?.as_bytes())?;

        upload_to_s3(
            &s3_client,
            &extraction_item.output_location,
            &output_temp_file.path(),
        )
        .await?;

        if temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(temp_file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
            }
        }
        if output_temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(output_temp_file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
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
                println!("Task failed");
                log_task(
                    task_id.clone(),
                    Status::Failed,
                    Some("Task failed".to_string()),
                    Some(Utc::now()),
                    &pg_pool,
                )
                .await?;
            }
            Err(e)
        }
    }
}
