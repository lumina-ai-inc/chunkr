use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::{ ExtractionPayload,  SegmentationStrategy };
use crate::models::server::segment::{PdlaSegment, Segment};
use crate::models::server::task::Status;
use crate::utils::configs::s3_config::create_client;
use crate::utils::configs::worker_config::Config as WorkerConfig;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::services::{
    log::log_task, payload::produce_extraction_payloads, pdf::{split_pdf, extract_text_pdf}, pdla::pdla_extraction,
};
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::Utc;
use std::io::Write;
use std::path::PathBuf;
use tempdir::TempDir;
use tempfile::NamedTempFile;
use lopdf::Object;
use uuid::Uuid;
use crate::models::server::segment::{ SegmentType, BoundingBox };

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("1");
    println!("Processing task");
    println!("2");
    let s3_client = create_client().await?;
    println!("3");
    let reqwest_client = reqwest::Client::new();
    println!("4");
    let extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    println!("5");
    let task_id = extraction_payload.task_id.clone();
    println!("6");
    let pg_pool = create_pool();
    println!("7");
    let extraction_config = WorkerConfig::from_env().unwrap();
    println!("8");

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        println!("9");
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Segmentation started".to_string()),
            None,
            &pg_pool,
        )
        .await?;
        println!("10");

        let pdf_file = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_payload.pdf_location,
            None,
        )
        .await?;
        println!("11");

        let mut split_temp_files: Vec<PathBuf> = Vec::new();
        println!("12");
        let split_temp_dir = TempDir::new("split_pdf")?;
        println!("13");

        if let Some(batch_size) = extraction_payload.batch_size {
            split_temp_files =
                split_pdf(pdf_file.path(), batch_size as usize, split_temp_dir.path())?;
        } else {
            split_temp_files.push(pdf_file.path().to_path_buf());
        }
        println!("14");

        let mut combined_segments: Vec<Segment> = Vec::new();
        let mut page_offset: u32 = 0;
        let mut batch_number: i32 = 0;
        println!("15");

        for temp_file in &split_temp_files {
            println!("16");
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
            println!("17");

            log_task(
                task_id.clone(),
                Status::Processing,
                Some(segmentation_message),
                None,
                &pg_pool,
            )
            .await?;
            println!("18");

            let temp_file_path = temp_file.to_path_buf();
            println!("19");

            match extraction_payload.configuration.segmentation_strategy {
                Some(SegmentationStrategy::Page) => {
                    println!("20");
                    let mut segments = Vec::new();
                    let page_texts = extract_text_pdf(&temp_file_path).await?;
                    let doc = lopdf::Document::load(&temp_file_path)?;

                    for (page_num, obj_id) in doc.get_pages() {
                        if let Ok(page_dict) = doc.get_dictionary(obj_id) {
                            if let Ok(mediabox) = page_dict.get(b"MediaBox").and_then(Object::as_array) {
                                if mediabox.len() >= 4 {
                                    let x1 = mediabox[0].as_float().unwrap_or(0.0);
                                    let y1 = mediabox[1].as_float().unwrap_or(0.0);
                                    let x2 = mediabox[2].as_float().unwrap_or(0.0);
                                    let y2 = mediabox[3].as_float().unwrap_or(0.0);
                                    
                                    let width = (x2 - x1).abs();
                                    let height = (y2 - y1).abs();
                                    let content = page_texts[(page_num - 1) as usize].clone();
                                    let mut segment = Segment {
                                        segment_id: Uuid::new_v4().to_string(),
                                        content: content,
                                        bbox: BoundingBox {
                                            top: 0.0,
                                            left: 0.0,
                                            width,
                                            height,
                                        },
                                        page_number: (page_num) as u32,
                                        page_width: width,
                                        page_height: height,
                                        segment_type: SegmentType::Page,
                                        image: None,
                                        html: None,
                                        markdown: None,
                                        ocr: None,
                                    };
                                    segment.finalize();
                                    segments.push(segment);
                                }
                            }
                        }
                    }
                    println!("21");

                    for item in &mut segments {
                        item.page_number += page_offset;
                    }
                    combined_segments.extend(segments);
                    page_offset += extraction_payload.batch_size.unwrap_or(1) as u32;
                }
                _ => {
                    println!("22");
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
            }
        }
        println!("23");

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&combined_segments)?.as_bytes())?;

        upload_to_s3(
            &s3_client,
            &extraction_payload.output_location,
            output_temp_file.path(),
        )
        .await?;
        println!("24");

        if output_temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(output_temp_file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
            }
        }

        Ok(())
        })
    .await;

    println!("25");
    match result {
        Ok(_) => {
            println!("Task succeeded");
            let extraction_config = WorkerConfig::from_env()?;
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
            .await
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
