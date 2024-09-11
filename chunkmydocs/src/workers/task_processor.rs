use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::{DateTime, Utc};
use chunkmydocs::extraction::pdf2png::convert_pdf_to_png;
use chunkmydocs::extraction::pdf2png::BoundingBox;
use chunkmydocs::extraction::pdla::pdla_extraction;
use chunkmydocs::extraction::table_ocr::table_extraction_from_image;
use chunkmydocs::models::rrq::queue::QueuePayload;
use chunkmydocs::models::server::extract::ExtractionPayload;
use chunkmydocs::models::server::extract::{TableOcr, TableOcrModel};
use chunkmydocs::models::server::segment::SegmentType;
use chunkmydocs::models::server::segment::{Chunk, Segment};
use chunkmydocs::models::server::task::Status;
use chunkmydocs::utils::configs::extraction_config;
use chunkmydocs::utils::db::deadpool_postgres;
use chunkmydocs::utils::db::deadpool_postgres::{Client, Pool};
use chunkmydocs::utils::json2mkd::json_2_mkd::hierarchical_chunk_and_add_markdown;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::storage::config_s3::create_client;
use chunkmydocs::utils::storage::services::{download_to_tempfile, upload_to_s3};
use std::path::Path;
use tempfile::TempDir;
use uuid::Uuid;

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

async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_item: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_item.task_id.clone();

    let pg_pool = deadpool_postgres::create_pool();

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

        let output_path = pdla_extraction(
            temp_file.path(),
            extraction_item.model,
            extraction_item.batch_size,
        )
        .await?;

        let file_content = tokio::fs::read_to_string(&output_path).await?;
        let segments: Vec<Segment> = serde_json::from_str(&file_content)?;

        let mut chunks =
            hierarchical_chunk_and_add_markdown(segments, extraction_item.target_chunk_length)
                .await?;
        if let Some(table_ocr_option) = extraction_item.table_ocr {
            chunks = table_ocr(chunks, temp_file.path(), Some(table_ocr_option)).await?;
        }
        let chunked_content = serde_json::to_string(&chunks)?;
        tokio::fs::write(&output_path, chunked_content).await?;

        upload_to_s3(&s3_client, &extraction_item.output_location, &output_path).await?;

        if temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(temp_file.path()) {
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

pub async fn table_ocr(
    chunks: Vec<Chunk>,
    pdf_path: &Path,
    table_ocr: Option<TableOcr>,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let output_format = if table_ocr.is_some() {
        TableOcr::HTML
    } else {
        TableOcr::JSON
    };

    let mut modified_chunks = Vec::new();
    let mut bounding_boxes = Vec::new();

    // First, collect all table segments and create bounding boxes
    for chunk in &chunks {
        for segment in &chunk.segments {
            if segment.segment_type == SegmentType::Table {
                bounding_boxes.push(BoundingBox {
                    left: segment.left,
                    top: segment.top,
                    width: segment.width,
                    height: segment.height,
                    page_number: segment.page_number,
                    bb_id: Uuid::new_v4().to_string(),
                });
            }
        }
    }

    // Convert PDF to PNG for all table segments at once
    let response = convert_pdf_to_png(pdf_path, bounding_boxes.clone()).await?;

    // Process chunks and segments
    for chunk in chunks {
        let mut modified_segments = Vec::new();
        for segment in chunk.segments {
            if segment.segment_type == SegmentType::Table {
                // Find the corresponding PNG page
                if let Some(png_page) = response.png_pages.iter().find(|p| {
                    let bb_id = bounding_boxes
                        .iter()
                        .find(|bb| {
                            bb.left == segment.left
                                && bb.top == segment.top
                                && bb.width == segment.width
                                && bb.height == segment.height
                                && bb.page_number == segment.page_number
                        })
                        .map(|bb| bb.bb_id.as_str());
                    p.bb_id == bb_id.unwrap_or_default()
                }) {
                    let temp_dir = TempDir::new()?;
                    let temp_file_path = temp_dir.path().join(format!("{}.png", png_page.bb_id));
                    let image_data = STANDARD.decode(&png_page.base64_png)?;
                    std::fs::write(&temp_file_path, &image_data)?;
                    let output = table_extraction_from_image(
                        &temp_file_path,
                        TableOcrModel::EasyOcr,
                        output_format,
                    )
                    .await?;
                    let modified_segment = Segment {
                        text: output,
                        ..segment
                    };
                    modified_segments.push(modified_segment);
                } else {
                    modified_segments.push(segment);
                }
            } else {
                modified_segments.push(segment);
            }
        }
        let modified_chunk = Chunk {
            segments: modified_segments,
            ..chunk
        };
        modified_chunks.push(modified_chunk);
    }

    Ok(modified_chunks)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;
    println!("Starting task processor");
    consumer(process, config.extraction_queue, 1, 600).await?;
    Ok(())
}
