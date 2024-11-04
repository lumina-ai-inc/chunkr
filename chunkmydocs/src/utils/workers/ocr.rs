use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::{ExtractionPayload, OcrStrategy};
use crate::models::server::segment::{OutputResponse, Segment, SegmentType};
use crate::models::server::task::Status;
use crate::utils::configs::extraction_config;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::services::{
    log::log_task,
    ocr::{download_and_ocr, download_and_table_ocr},
    payload::produce_extraction_payloads,
};
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::Utc;
use futures::future::try_join_all;
use std::{
    fs::File,
    io::{BufReader, Write},
    sync::Arc,
};
use tempfile::NamedTempFile;
use tokio::sync::Semaphore;

pub fn filter_segment(segment: &Segment, ocr_strategy: OcrStrategy) -> bool {
    if segment.image.is_none() {
        return false;
    }
    match ocr_strategy {
        OcrStrategy::Off => false,
        OcrStrategy::All => true,
        OcrStrategy::Auto => match segment.segment_type {
            SegmentType::Table => true,
            SegmentType::Picture => true,
            _ => {
                if segment.content.is_empty() {
                    true
                } else {
                    false
                }
            }
        },
    }
}

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_payload.task_id.clone();
    let pg_pool = create_pool();
    let config = extraction_config::Config::from_env()?;

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("OCR started".to_string()),
            None,
            &pg_pool,
        )
        .await?;

        let output_file: NamedTempFile = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_payload.output_location,
            None,
        )
        .await?;

        let file = File::open(output_file.path())?;
        let reader = BufReader::new(file);
        let mut output_response: OutputResponse = serde_json::from_reader(reader)?;

        let semaphore = Arc::new(Semaphore::new(config.ocr_concurrency));

        try_join_all(output_response.chunks.iter_mut().flat_map(|chunk| {
            chunk.segments.iter_mut().filter_map(|segment| {
                if filter_segment(
                    segment,
                    extraction_payload.configuration.ocr_strategy.clone(),
                ) {
                    Some(async {
                        let semaphore = semaphore.clone();
                        let s3_client = s3_client.clone();
                        let reqwest_client = reqwest_client.clone();
                        let _permit = semaphore.acquire().await?;
                        let ocr_result = if segment.segment_type == SegmentType::Table {
                            download_and_table_ocr(
                                &s3_client,
                                &reqwest_client,
                                &segment.image.as_ref().unwrap(),
                            )
                            .await
                        } else {
                            download_and_ocr(
                                &s3_client,
                                &reqwest_client,
                                &segment.image.as_ref().unwrap(),
                            )
                            .await
                        };
                        drop(_permit);
                        match ocr_result {
                            Ok((ocr_result, html, markdown)) => {
                                segment.ocr = Some(ocr_result);
                                segment.finalize();
                                if !html.is_empty() {
                                    segment.html = Some(html);
                                }
                                if !markdown.is_empty() {
                                    segment.markdown = Some(markdown);
                                }
                                Ok::<_, Box<dyn std::error::Error>>(())
                            }
                            Err(e) => {
                                eprintln!("Error processing OCR: {:?}", e);
                                Ok::<_, Box<dyn std::error::Error>>(())
                            }
                        }
                    })
                } else {
                    None
                }
            })
        }))
        .await?;

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
            if extraction_payload.configuration.json_schema.is_some() {
                produce_extraction_payloads(
                    config.queue_structured_extract,
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
