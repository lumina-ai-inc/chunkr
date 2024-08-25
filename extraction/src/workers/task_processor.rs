use chrono::Utc;
use chunkmydocs::extraction::grobid::grobid_extraction;
use chunkmydocs::extraction::pdla::pdla_extraction;
use chunkmydocs::models::extraction::extract::ExtractionPayload;
use chunkmydocs::models::extraction::extract::ModelInternal;
use chunkmydocs::models::extraction::segment::{Chunk, Segment, SegmentType};
use chunkmydocs::models::extraction::task::Status;
use chunkmydocs::models::rrq::{produce::ProducePayload, queue::QueuePayload};
use chunkmydocs::utils::configs::extraction_config;
use chunkmydocs::utils::db::deadpool_postgres;
use chunkmydocs::utils::db::deadpool_postgres::{Client, Pool};
use chunkmydocs::utils::json2mkd::json2mkd::chunk_and_add_markdown;
use chunkmydocs::utils::rrq::{consumer::consumer, service::produce};
use chunkmydocs::utils::storage_service::services::{download_to_tempfile, upload_to_s3};
use diesel::pg;
use humantime::format_duration;
use serde_json::json;
use std::{fs, path::PathBuf};
use uuid::Uuid;

pub async fn log_task(
    task_id: String,
    file_id: String,
    status: Status,
    message: Option<String>,
    finished_at: Option<String>,
    pool: &Pool,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;

    println!("Prepared status: {:?}", status);
    println!("Prepared task_id: {}", task_id);
    println!("Prepared file_id: {}", file_id);

    let mut client: Client = pool.get().await?;

    let task_query = format!(
        "UPDATE ingestion_tasks SET status = '{:?}', message = '{}', finished_at = '{:?}' WHERE task_id = '{}'",
        status,
        message.unwrap_or_default(),
        finished_at.unwrap_or_default(),
        task_id
    );

    client.execute(&task_query, &[]).await?;

    let files_query = format!(
        "UPDATE ingestion_files SET status = '{:?}' WHERE task_id = '{}' AND file_id = '{}'",
        status, task_id, file_id
    );

    client.execute(&files_query, &[]).await?;

    Ok(())
}

async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    let extraction_item: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_item.task_id.clone();
    let file_id = extraction_item.file_id.clone();

    let pg_pool = deadpool_postgres::create_pool();

    println!("{:?}", extraction_item.clone());
    // Import the necessary types if they're not already imported at the top of the file

    // Get the database pool from the configuration

    // Pass the pool to the log_task function
    log_task(
        task_id.clone(),
        file_id.clone(),
        Status::Processing,
        Some(format!(
            "Task processing | Retry ({}/{})",
            payload.attempt, payload.max_attempts
        )),
        None,
        &pg_pool,
    )
    .await?;

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        let temp_file = download_to_tempfile(&extraction_item.input_location).await?;
        println!("Downloaded file to {:?}", temp_file.path());

        let output_path: PathBuf;
        let chunks: Vec<Chunk>;

        if extraction_item.model == ModelInternal::Grobid {
            output_path = grobid_extraction(temp_file.path()).await?;
            // TODO: Implement chunk_and_add_markdown for Grobid output if needed
        } else if extraction_item.model == ModelInternal::Pdla
            || extraction_item.model == ModelInternal::PdlaFast
        {
            output_path = pdla_extraction(
                temp_file.path(),
                extraction_item.model,
                extraction_item.batch_size,
            )
            .await?;

            // Read the PDLA output file
            let file_content = tokio::fs::read_to_string(&output_path).await?;
            let segments: Vec<Segment> = serde_json::from_str(&file_content)?;

            // Apply chunk_and_add_markdown
            chunks = chunk_and_add_markdown(segments, 512).await?;

            // Write the chunked and markdown-added content back to the file
            let chunked_content = serde_json::to_string(&chunks)?;
            tokio::fs::write(&output_path, chunked_content).await?;
        } else {
            return Err("Invalid model".into());
        }

        upload_to_s3(
            &extraction_item.output_location,
            output_path.clone().to_str().unwrap(),
            fs::read(output_path)?,
            extraction_item
                .expiration
                .map(|d| format_duration(d).to_string())
                .as_deref(),
        )
        .await?;

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
            log_task(
                task_id.clone(),
                file_id.clone(),
                Status::Succeeded,
                Some("Task succeeded".to_string()),
                Some(Utc::now().to_string()),
                &pg_pool,
            )
            .await?;
            println!("Task succeeded");
            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing task: {:?}", e);
            if payload.attempt >= payload.max_attempts {
                log_task(
                    task_id.clone(),
                    file_id.clone(),
                    Status::Failed,
                    Some(e.to_string()),
                    Some(Utc::now().to_string()),
                    &pg_pool,
                )
                .await?;
                println!("Task failed");
            }
            Err(e)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;
    consumer(process, config.extraction_queue, 1, 600).await?;
    Ok(())
}

pub async fn process_bounding_boxes(
    file_path: &str,
    target_size: usize,
) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    println!("Processing file: {}", file_path);
    let file_content = tokio::fs::read_to_string(file_path).await?;
    println!("File content loaded, length: {}", file_content.len());

    let mut segments: Vec<Segment> = serde_json::from_str(&file_content)?;
    println!("Parsed {} segments", segments.len());
    println!("Segment types processed");
    chunk_and_add_markdown(segments, target_size).await
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_process_bounding_boxes() -> Result<(), Box<dyn std::error::Error>> {
        // Load the bounding_boxes.json file
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push(
            "/Users/ishaankapoor/chunk-my-docs/example/output/00c08086-9837-5551-8133-4e22ac28c6a5-HighQuality/bounding_boxes.json",
        );
        let file_path = path.to_str().unwrap();

        // Process the bounding boxes
        let chunks = process_bounding_boxes(file_path, 512).await?;

        println!("{:?}", chunks);
        Ok(())
    }
}
