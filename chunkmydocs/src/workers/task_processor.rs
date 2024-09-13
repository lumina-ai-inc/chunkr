use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::{DateTime, Utc};
use chunkmydocs::extraction::llm::apply_llm;
use chunkmydocs::extraction::pdf2png::convert_pdf_to_png;
use chunkmydocs::extraction::pdf2png::BoundingBox;
use chunkmydocs::extraction::pdla::pdla_extraction;
use chunkmydocs::extraction::table_ocr::table_extraction_from_image;
use chunkmydocs::models::rrq::queue::QueuePayload;
use chunkmydocs::models::server::extract::ExtractionPayload;
use chunkmydocs::models::server::extract::PipelinePayload;
use chunkmydocs::models::server::extract::{TableOcr, TableOcrModel};
use chunkmydocs::models::server::llm::LLMConfig;
use chunkmydocs::models::server::segment::PngPage;
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
        let segments_raw: Vec<serde_json::Value> = serde_json::from_str(&file_content)?;
        let mut segments: Vec<Segment> = segments_raw
            .into_iter()
            .enumerate()
            .map(|(index, mut segment_value)| {
                let segment_id = format!("seg{}_uuid", index + 1);
                segment_value["segment_id"] = serde_json::Value::String(segment_id);
                serde_json::from_value(segment_value).unwrap()
            })
            .collect();
        //transform - or OCR here?
        //LLMs here?
        if let Some(pipeline) = extraction_item.pipeline {
            segments = apply_pipeline(segments.clone(), pipeline, temp_file.path()).await?;
        }
        let mut chunks =
            hierarchical_chunk_and_add_markdown(segments, extraction_item.target_chunk_length)
                .await?;

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

async fn apply_pipeline(
    segments: Vec<Segment>,
    config: PipelinePayload,
    pdf_path: &Path,
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let bounding_boxes = collect_bounding_boxes(&segments, &config);
    let png_pages_resp = convert_pdf_to_png(pdf_path, bounding_boxes).await?;
    let png_pages: Vec<PngPage> = png_pages_resp
        .png_pages
        .into_iter()
        .map(|p| p.into())
        .collect();

    let mut segments = segments;

    if config.table_ocr.is_some() {
        segments = apply_table_ocr(segments, &png_pages, config.table_ocr).await?;
    }

    if let Some(llm_config) = config.llm_model {
        segments = apply_llm_to_segments(segments, llm_config, &png_pages).await?;
    }

    Ok(segments)
}

fn collect_bounding_boxes(segments: &[Segment], config: &PipelinePayload) -> Vec<BoundingBox> {
    segments
        .iter()
        .filter(|s| {
            (s.segment_type == SegmentType::Picture && config.llm_model.is_some())
                || (s.segment_type == SegmentType::Table
                    && (config.table_ocr.is_some() || config.llm_model.is_some()))
        })
        .map(|s| BoundingBox {
            left: s.left,
            top: s.top,
            width: s.width,
            height: s.height,
            page_number: s.page_number,
            bb_id: Uuid::new_v4().to_string(),
        })
        .collect()
}

async fn apply_table_ocr(
    segments: Vec<Segment>,
    png_pages: &[PngPage],
    table_ocr: Option<TableOcr>,
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let output_format = table_ocr.unwrap_or(TableOcr::JSON);

    let mut result = Vec::new();

    for segment in segments {
        if segment.segment_type == SegmentType::Table {
            if let Some(png_page) = png_pages.iter().find(|p| p.bb_id == segment.segment_id) {
                let temp_dir = TempDir::new()?;
                let temp_file_path = temp_dir.path().join(format!("{}.png", png_page.bb_id));
                let image_data = STANDARD.decode(&png_page.base64_png)?;
                std::fs::write(&temp_file_path, &image_data)?;
                let ocr_output = table_extraction_from_image(
                    &temp_file_path,
                    TableOcrModel::EasyOcr,
                    output_format,
                )
                .await?;
                result.push(Segment {
                    text: ocr_output,
                    ..segment
                });
            } else {
                result.push(segment);
            }
        } else {
            result.push(segment);
        }
    }

    Ok(result)
}

async fn apply_llm_to_segments(
    segments: Vec<Segment>,
    llm_config: LLMConfig,
    png_pages: &[PngPage],
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let table_prompt =
        "Extract all the tables from the image and return the data in a markdown table";
    let picture_prompt = "Describe the image";

    let extraction_config = extraction_config::Config::from_env()?;

    let mut result = Vec::new();
    for segment in segments {
        let prompt = if segment.segment_type == SegmentType::Table {
            table_prompt.to_string()
        } else {
            picture_prompt.to_string()
        };
        let llm_output = apply_llm(
            llm_config.clone(),
            &extraction_config,
            png_pages.to_vec(),
            prompt.clone(),
        )
        .await?;

        result.push(Segment {
            text: format!(
                "Table: {}\n\n LLM Table Output:\n{}",
                segment.text, llm_output
            ),
            ..segment
        });
    }
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;
    println!("Starting task processor");
    consumer(process, config.extraction_queue, 1, 600).await?;
    Ok(())
}
