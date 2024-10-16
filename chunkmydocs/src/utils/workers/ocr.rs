use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::{ ExtractionPayload, OcrStrategy };
use crate::models::server::segment::{ Chunk, Segment, SegmentType };
use crate::models::server::task::Status;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::services::rapid_ocr;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{ download_to_tempfile, upload_to_s3 };
use crate::utils::workers::log::log_task;
use chrono::Utc;
use futures::stream::{ self, StreamExt };
use reqwest::multipart;
use std::io::{ Read, Write };
use tempfile::NamedTempFile;

pub fn filter_segment(segment: &Segment, ocr_strategy: OcrStrategy) -> bool {
    match ocr_strategy {
        OcrStrategy::Off => false,
        OcrStrategy::All => true,
        OcrStrategy::Auto => {
            match segment.segment_type {
                SegmentType::Table => true,
                SegmentType::Picture => true,
                _ => {
                    if segment.content.is_empty() { true } else { false }
                }
            }
        }
    }
}

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_payload.task_id.clone();
    let pg_pool = create_pool();

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("OCR started".to_string()),
            None,
            &pg_pool
        ).await?;

        let chunks_file: NamedTempFile = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_payload.output_location,
            None
        ).await?;

        let mut file_contents = String::new();
        chunks_file.as_file().read_to_string(&mut file_contents)?;
        let chunks: Vec<Chunk> = match serde_json::from_str(&file_contents) {
            Ok(segments) => segments,
            Err(e) => {
                return Err(Box::new(e) as Box<dyn std::error::Error>);
            }
        };

        let filtered_segments: Vec<&Segment> = chunks
            .iter()
            .flat_map(|chunk| chunk.segments.iter())
            .filter(|segment|
                filter_segment(segment, extraction_payload.configuration.ocr_strategy.clone())
            )
            .collect();

        // let results: Vec<(Segment, serde_json::Value)> = stream::iter(filtered_segments).for_each_concurrent(None, |segment| async {
        //     let result: Result<(Segment, serde_json::Value), Box<dyn std::error::Error>> = (async {
        //         let cropped_segment_location = format!(
        //             "{}/{}",
        //             extraction_payload.image_folder_location,
        //             segment.segment_id
        //         );
        //         let cropped_image = download_to_tempfile(
        //             &s3_client,
        //             &reqwest_client,
        //             &cropped_segment_location,
        //             None
        //         ).await?;

        //         let rapid_ocr_response: serde_json::Value = rapid_ocr::call_rapid_ocr_api(
        //             &cropped_image.path()
        //         ).await?;

        //         Ok((segment.clone(), rapid_ocr_response))
        //     }).await;

        //     if let Err(e) = result {
        //         eprintln!("Error processing segment: {:?}", e);
        //     }
        // }).collect();

        Ok(())
    }).await;

    match result {
        Ok(_) => {
            println!("Task succeeded");
            log_task(
                task_id.clone(),
                Status::Succeeded,
                Some("Task succeeded".to_string()),
                Some(Utc::now()),
                &pg_pool
            ).await?;
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
                    &pg_pool
                ).await?;
            }
            Err(e)
        }
    }
}
