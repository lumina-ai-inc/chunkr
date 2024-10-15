use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::{ ExtractionPayload, OcrStrategy };
use crate::models::server::segment::{ PdlaSegment, Segment };
use crate::models::server::task::Status;
use crate::utils::configs::extraction_config::Config;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::json2mkd::json_2_mkd::hierarchical_chunking;
use crate::utils::services::crop::crop_image;
use crate::utils::services::pdf::split_pdf;
use crate::utils::services::pdla::pdla_extraction;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{ download_to_tempfile, upload_to_s3 };
use crate::utils::workers::{ log::log_task, payload::produce_extraction_payloads };
use chrono::Utc;
use futures::future::try_join_all;
use rayon::prelude::*;
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

        let mut combined_segments: Vec<Segment> = Vec::new();
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
            combined_segments.extend(segments);
            page_offset += extraction_payload.batch_size.unwrap_or(1) as u32;
        }

        let image_folder_location = extraction_payload.image_folder_location.clone();
        let page_image_paths = try_join_all(
            (0..extraction_payload.page_count.unwrap()).map(|i| {
                let image_folder = image_folder_location.clone();
                let s3_client = s3_client.clone();
                let reqwest_client = reqwest_client.clone();
                async move {
                    let image_name = format!("page_{}.jpg", i);
                    let image_location = format!("{}/{}", image_folder, image_name);
                    download_to_tempfile(&s3_client, &reqwest_client, &image_location, None).await
                }
            })
        ).await?;

        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Cropping segments".to_string()),
            None,
            &pg_pool
        ).await?;

        let cropped_temp_dir = TempDir::new("cropped_images")?;
        combined_segments.par_iter().for_each(|segment| {
            let page_index = segment.page_number as usize;
            if let Some(image_path) = page_image_paths.get(page_index) {
                crop_image(
                    &image_path.path().to_path_buf(),
                    segment,
                    &cropped_temp_dir.path().to_path_buf()
                );
            } else {
                eprintln!("Image not found for page {}", segment.page_number);
            }
        });

        for entry in std::fs::read_dir(cropped_temp_dir.path())? {
            let entry = entry?;
            let file_name = entry.file_name();
            let s3_key = format!(
                "{}/{}",
                extraction_payload.image_folder_location,
                file_name.to_string_lossy()
            );
            upload_to_s3(&s3_client, &s3_key, &entry.path()).await?;
        }

        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Chunking segments".to_string()),
            None,
            &pg_pool
        ).await?;

        let chunks = hierarchical_chunking(
            combined_segments,
            extraction_payload.target_chunk_length
        ).await?;

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&chunks)?.as_bytes())?;

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
            if extraction_payload.configuration.ocr_strategy == OcrStrategy::Off {
                log_task(
                    task_id.clone(),
                    Status::Succeeded,
                    Some("Task succeeded".to_string()),
                    Some(Utc::now()),
                    &pg_pool
                ).await?;
            } else {
                let config = Config::from_env()?;
                produce_extraction_payloads(config.queue_ocr, extraction_payload).await?;
            }
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
