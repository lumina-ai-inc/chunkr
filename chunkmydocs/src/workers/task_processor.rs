use chrono::{ DateTime, Utc };
use std::path::Path;
use chunkmydocs::extraction::pdf2png::{ convert_pdf_to_png, BoundingBox };
use chunkmydocs::extraction::pdla::pdla_extraction;
use chunkmydocs::models::rrq::queue::QueuePayload;
use chunkmydocs::models::server::extract::{ ExtractionPayload, Configuration };
use chunkmydocs::models::server::segment::{ PngPage, Segment };
use chunkmydocs::models::server::task::Status;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::configs::extraction_config;
use chunkmydocs::utils::db::deadpool_postgres::{ Client, Pool, create_pool };
use chunkmydocs::utils::json2mkd::json_2_mkd::hierarchical_chunk_and_add_markdown;
use chunkmydocs::utils::storage::config_s3::create_client;
use chunkmydocs::utils::storage::services::{ download_to_tempfile, upload_to_s3 };

pub async fn log_task(
    task_id: String,
    status: Status,
    message: Option<String>,
    finished_at: Option<DateTime<Utc>>,
    pool: &Pool
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

    let pg_pool = create_pool();

    log_task(
        task_id.clone(),
        Status::Processing,
        Some(format!("Task processing | Tries ({}/{})", payload.attempt, payload.max_attempts)),
        None,
        &pg_pool
    ).await?;

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        let temp_file = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_item.input_location,
            None
        ).await?;

        let output_path = pdla_extraction(
            temp_file.path(),
            extraction_item.model,
            extraction_item.batch_size
        ).await?;

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
        if let Some(config) = extraction_item.configuration {
            segments = apply_pipeline(segments.clone(), config, temp_file.path()).await?;
        }
        let chunks = hierarchical_chunk_and_add_markdown(
            segments,
            extraction_item.target_chunk_length
        ).await?;

        let chunked_content = serde_json::to_string(&chunks)?;
        tokio::fs::write(&output_path, chunked_content).await?;

        upload_to_s3(&s3_client, &extraction_item.output_location, &output_path).await?;

        if temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(temp_file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
            }
        }

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
                println!("Task failed");
                log_task(
                    task_id.clone(),
                    Status::Failed,
                    Some("Task failed".to_string()),
                    Some(Utc::now()),
                    &pg_pool
                ).await?;
            }
            Err(e)
        }
    }
}

async fn apply_pipeline(
    segments: Vec<Segment>,
    config: Configuration,
    pdf_path: &Path
) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let bounding_boxes = segments_to_bounding_boxes(&segments);
    let png_pages_resp = convert_pdf_to_png(pdf_path, bounding_boxes).await?;
    let png_pages: Vec<PngPage> = png_pages_resp.png_pages
        .into_iter()
        .map(|p| p.into())
        .collect();

    let mut segments = segments;

    // if let Some(llm_config) = config.LLM {
    //     // Filter segments based on affected_segments in llm_config
    //     let affected_types = &llm_config.affected_segments;
    //     let filtered_segments: Vec<Segment> = segments
    //         .iter()
    //         .filter(|s| affected_types.contains(&s.segment_type))
    //         .cloned()
    //         .collect();

    //     let processed_segments =
    //         apply_llm_to_segments(filtered_segments, llm_config.clone(), &png_pages).await?;

    //     // Replace the affected segments with processed ones
    //     segments = segments
    //         .into_iter()
    //         .map(|s| {
    //             if affected_types.contains(&s.segment_type) {
    //                 processed_segments
    //                     .iter()
    //                     .find(|ps| ps.segment_id == s.segment_id)
    //                     .cloned()
    //                     .unwrap_or(s)
    //             } else {
    //                 s
    //             }
    //         })
    //         .collect();
    // }
    // uncomment this when we add llm

    Ok(segments)
}

fn segments_to_bounding_boxes(segments: &[Segment]) -> Vec<BoundingBox> {
    segments
        .iter()
        // .filter(|s| {
        //     (s.segment_type == SegmentType::Picture && config.llm_model.is_some())
        //         || (s.segment_type == SegmentType::Table
        //             && (config.table_ocr.is_some() || config.llm_model.is_some()))
        // })
        .map(|s| BoundingBox {
            left: s.left,
            top: s.top,
            width: s.width,
            height: s.height,
            page_number: s.page_number,
            bb_id: s.segment_id.clone(),
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;
    println!("Starting task processor");
    consumer(process, config.extraction_queue, 1, 600).await?;
    Ok(())
}
