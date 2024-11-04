use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::{ExtractionPayload, OcrStrategy};
use crate::models::server::segment::{Chunk, OutputResponse, Segment};
use crate::models::server::task::Status;
use crate::utils::configs::extraction_config::Config as ExtractionConfig;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::services::{
    chunking::hierarchical_chunking, images::crop_image, log::log_task,
    payload::produce_extraction_payloads,
};
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::Utc;
use futures::future::try_join_all;
use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufReader, Write},
};
use tempdir::TempDir;
use tempfile::NamedTempFile;

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_payload.task_id.clone();
    let pg_pool = create_pool();

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Chunking started".to_string()),
            None,
            &pg_pool,
        )
        .await?;

        let segments_file = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_payload.output_location,
            None,
        )
        .await?;

        let file = File::open(segments_file.path())?;
        let reader = BufReader::new(file);
        let mut segments: Vec<Segment> = serde_json::from_reader(reader)?;

        let image_folder_location = extraction_payload.image_folder_location.clone();
        let page_image_files = try_join_all((0..extraction_payload.page_count.unwrap()).map(|i| {
            let image_folder = image_folder_location.clone();
            let s3_client = s3_client.clone();
            let reqwest_client = reqwest_client.clone();
            async move {
                let image_name = format!("page_{}.jpg", i + 1);
                let image_location = format!("{}/{}", image_folder, image_name);
                download_to_tempfile(&s3_client, &reqwest_client, &image_location, None).await
            }
        }))
        .await?;

        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Cropping segments".to_string()),
            None,
            &pg_pool,
        )
        .await?;

        let cropped_temp_dir = TempDir::new("cropped_images")?;
        segments.par_iter().for_each(|segment| {
            let page_index = (segment.page_number as usize) - 1;
            if let Some(image_path) = page_image_files.get(page_index) {
                let crop_result = crop_image(
                    &image_path.path().to_path_buf(),
                    segment,
                    &cropped_temp_dir.path().to_path_buf(),
                )
                .map_err(|e| {
                    format!(
                        "Failed to crop image for segment {} on page {}: {:?}",
                        segment.segment_id, segment.page_number, e
                    )
                });
                match crop_result {
                    Ok(_) => (),
                    Err(err_msg) => eprintln!("{}", err_msg),
                }
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
            segments
                .iter_mut()
                .find(|segment| {
                    segment.segment_id
                        == file_name
                            .clone()
                            .to_string_lossy()
                            .split(".")
                            .next()
                            .unwrap()
                })
                .map(|segment| {
                    segment.image = Some(s3_key.clone());
                });
        }

        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Chunking segments".to_string()),
            None,
            &pg_pool,
        )
        .await?;

        let chunks: Vec<Chunk> =
            hierarchical_chunking(segments, extraction_payload.target_chunk_length).await?;

        let output_response = OutputResponse {
            chunks,
            extracted_json: None,
        };

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&output_response)?.as_bytes())?;

        upload_to_s3(
            &s3_client,
            &extraction_payload.output_location,
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
            if extraction_payload.configuration.ocr_strategy == OcrStrategy::Off {
                if extraction_payload.configuration.json_schema.is_some() {
                    let extraction_config = ExtractionConfig::from_env()?;
                    produce_extraction_payloads(
                        extraction_config.queue_structured_extract,
                        extraction_payload.clone(),
                    )
                    .await?;
                    log_task(
                        task_id.clone(),
                        Status::Processing,
                        Some("Structured extraction queued".to_string()),
                        None,
                        &pg_pool,
                    )
                    .await?;
                } else {
                    log_task(
                        task_id.clone(),
                        Status::Succeeded,
                        Some("Task succeeded".to_string()),
                        Some(Utc::now()),
                        &pg_pool,
                    )
                    .await?;
                }
            } else {
                let extraction_config = ExtractionConfig::from_env()?;
                produce_extraction_payloads(
                    extraction_config.queue_ocr,
                    extraction_payload.clone(),
                )
                .await?;
                log_task(
                    task_id.clone(),
                    Status::Processing,
                    Some("OCR queued".to_string()),
                    None,
                    &pg_pool,
                )
                .await?;
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
                    Some("Chunking failed".to_string()),
                    Some(Utc::now()),
                    &pg_pool,
                )
                .await?;
            }
            Err(e)
        }
    }
}
