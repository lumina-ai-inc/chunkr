use crate::models::rrq::produce::ProducePayload;
use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::ExtractionPayload;
use crate::models::server::segment::{PdlaSegment, Segment};
use crate::models::server::task::Status;
use crate::utils::configs::extraction_config::Config;
use crate::utils::db::deadpool_postgres::{create_pool, Client, Pool};
use crate::utils::rrq::service::produce;
use crate::utils::services::pdf::split_pdf;
use crate::utils::services::pdla::pdla_extraction;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::{DateTime, Utc};
use std::io::Write;
use std::path::PathBuf;
use tempdir::TempDir;
use tempfile::NamedTempFile;


pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client: aws_sdk_s3::Client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_item: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_item.task_id.clone();
    let user_id = extraction_item.user_id.clone();
    let pg_pool = create_pool();
    let client: Client = pg_pool.get().await?;
    let file_name_query = "SELECT file_name FROM tasks WHERE task_id = $1 AND user_id = $2";
    let file_name_row = client
        .query_one(file_name_query, &[&task_id, &user_id])
        .await?;
    let file_name: String = file_name_row.get(0);

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
        let (final_output_path, s3_pdf_location, page_count, extension) = preprocess(
            &s3_client,
            &reqwest_client,
            &extraction_item,
            &task_id,
            &user_id,
            &client,
        )
        .await?;

        let mut split_temp_files: Vec<PathBuf> = Vec::new();
        let split_temp_dir = TempDir::new("split_pdf")?;

        if let Some(batch_size) = extraction_item.batch_size {
            split_temp_files = split_pdf(
                &final_output_path,
                batch_size as usize,
                split_temp_dir.path(),
            )
            .await?;
        } else {
            split_temp_files.push(final_output_path.clone());
        }

        let mut combined_output: Vec<Segment> = Vec::new();
        let mut page_offset: u32 = 0;
        let mut batch_number: i32 = 0;

        for temp_file in &split_temp_files {
            batch_number += 1;
            let segmentation_message = if split_temp_files.len() > 1 {
                format!(
                    "Segmenting | Batch {} of {}",
                    batch_number,
                    split_temp_files.len()
                )
            } else {
                "Segmenting".to_string()
            };

            log_task(
                task_id.clone(),
                Status::Processing,
                Some(segmentation_message),
                None,
                &pg_pool,
            )
            .await?;

            let temp_file_path = temp_file.to_path_buf();

            let pdla_response =
                pdla_extraction(&temp_file_path, extraction_item.model.clone()).await?;
            let pdla_segments: Vec<PdlaSegment> = serde_json::from_str(&pdla_response)?;
            let mut segments: Vec<Segment> = pdla_segments
                .iter()
                .map(|pdla_segment| pdla_segment.to_segment())
                .collect();

            for item in &mut segments {
                item.page_number += page_offset;
            }
            combined_output.extend(segments);
            page_offset += extraction_item.batch_size.unwrap_or(1) as u32;
        }

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&combined_output)?.as_bytes())?;
        println!(
            "Output file written: {:?}, Size: {} bytes",
            output_temp_file,
            std::fs::metadata(output_temp_file.path())?.len()
        );
        upload_to_s3(
            &s3_client,
            &extraction_item.output_location,
            &output_temp_file.path(),
        )
        .await?;

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

            produce_ocr_payload(extraction_item).await?;

            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing task: {:?}", e);
            let error_message = if e
                .to_string()
                .to_lowercase()
                .contains("usage limit exceeded")
            {
                "Task failed: Usage limit exceeded".to_string()
            } else {
                "Task failed".to_string()
            };

            if payload.attempt >= payload.max_attempts {
                eprintln!("Task failed after {} attempts", payload.max_attempts);
                log_task(
                    task_id.clone(),
                    Status::Failed,
                    Some(error_message),
                    Some(Utc::now()),
                    &pg_pool,
                )
                .await?;
            }
            Err(e)
        }
    }
}
