use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::ExtractionPayload;
use crate::models::server::segment::{PdlaSegment, Segment};
use crate::models::server::task::Status;
use crate::utils::configs::extraction_config::Config as ExtractionConfig;
use crate::utils::configs::s3_config::create_client;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::services::{
    log::log_task, payload::produce_extraction_payloads, pdf::split_pdf, pdla::pdla_extraction,
};
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::Utc;
use std::io::Write;
use std::path::PathBuf;
use tempdir::TempDir;
use tempfile::NamedTempFile;

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_payload.task_id.clone();
    let pg_pool = create_pool();
    let extraction_config = ExtractionConfig::from_env().unwrap();

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Segmentation started".to_string()),
            None,
            &pg_pool,
        )
        .await?;

        let pdf_file = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_payload.pdf_location,
            None,
        )
        .await?;

        let mut split_temp_files: Vec<PathBuf> = Vec::new();
        let split_temp_dir = TempDir::new("split_pdf")?;

        if let Some(batch_size) = extraction_payload.batch_size {
            split_temp_files =
                split_pdf(pdf_file.path(), batch_size as usize, split_temp_dir.path())?;
        } else {
            split_temp_files.push(pdf_file.path().to_path_buf());
        }

        let mut combined_segments: Vec<Segment> = Vec::new();
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
                pdla_extraction(&temp_file_path, extraction_payload.model.clone()).await?;
            let pdla_segments: Vec<PdlaSegment> = serde_json::from_str(&pdla_response)?;
            let mut segments: Vec<Segment> = pdla_segments
                .iter()
                .map(|pdla_segment| pdla_segment.to_segment())
                .map(|mut segment| {
                    let multipler =
                        extraction_config.page_image_density / extraction_config.pdf_density;
                    segment.bbox.top *= multipler;
                    segment.bbox.left *= multipler;
                    segment.bbox.width *= multipler;
                    segment.bbox.height *= multipler;
                    segment.page_height = (segment.page_height * multipler).round();
                    segment.page_width = (segment.page_width * multipler).round();
                    segment.bbox.width += extraction_config.segment_bbox_offset * 2.0;
                    segment.bbox.height += extraction_config.segment_bbox_offset * 2.0;
                    segment.bbox.left -= extraction_config.segment_bbox_offset;
                    segment.bbox.top -= extraction_config.segment_bbox_offset;
                    segment.finalize();
                    segment
                })
                .collect();

            for item in &mut segments {
                item.page_number += page_offset;
            }
            combined_segments.extend(segments);
            page_offset += extraction_payload.batch_size.unwrap_or(1) as u32;
        }

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&combined_segments)?.as_bytes())?;

        upload_to_s3(
            &s3_client,
            &extraction_payload.output_location,
            output_temp_file.path(),
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
            let extraction_config = ExtractionConfig::from_env()?;
            produce_extraction_payloads(
                extraction_config.queue_postprocess,
                extraction_payload.clone(),
            )
            .await?;
            log_task(
                task_id.clone(),
                Status::Processing,
                Some("Chunking queued".to_string()),
                None,
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
                    Some("Segmentation failed".to_string()),
                    Some(Utc::now()),
                    &pg_pool,
                )
                .await?;
            }
            Err(e)
        }
    }
}
