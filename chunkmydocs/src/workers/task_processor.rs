use chrono::Utc;
use chunkmydocs::extraction::grobid::grobid_extraction;
use chunkmydocs::extraction::pdla::pdla_extraction;
use chunkmydocs::models::extraction::extract::ExtractionPayload;
use chunkmydocs::models::extraction::extract::ModelInternal;
use chunkmydocs::models::extraction::segment::{Chunk, Segment};
use chunkmydocs::models::extraction::task::Status;
use chunkmydocs::models::rrq::queue::QueuePayload;
use chunkmydocs::utils::configs::extraction_config;
use chunkmydocs::utils::db::deadpool_postgres;
use chunkmydocs::utils::db::deadpool_postgres::{Client, Pool};
use chunkmydocs::utils::json2mkd::json_2_mkd::chunk_and_add_markdown;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::storage::config_s3::create_client;
use chunkmydocs::utils::storage::services::{download_to_tempfile, upload_to_s3};
use std::path::PathBuf;

pub async fn log_task(
    task_id: String,
    file_id: String,
    status: Status,
    message: Option<String>,
    finished_at: Option<String>,
    pool: &Pool,
) -> Result<(), Box<dyn std::error::Error>> {
    let client: Client = pool.get().await?;

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
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_item: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_item.task_id.clone();
    let file_id = extraction_item.file_id.clone();

    let pg_pool = deadpool_postgres::create_pool();

    log_task(
        task_id.clone(),
        file_id.clone(),
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

            let file_content = tokio::fs::read_to_string(&output_path).await?;
            let segments: Vec<Segment> = serde_json::from_str(&file_content)?;

            chunks = chunk_and_add_markdown(segments, 512).await?;

            let chunked_content = serde_json::to_string(&chunks)?;
            tokio::fs::write(&output_path, chunked_content).await?;
        } else {
            return Err("Invalid model".into());
        }

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
            log_task(
                task_id.clone(),
                file_id.clone(),
                Status::Succeeded,
                Some("Task succeeded".to_string()),
                Some(Utc::now().to_string()),
                &pg_pool,
            )
            .await?;
            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing task: {:?}", e);
            if payload.attempt >= payload.max_attempts {
                log_task(
                    task_id.clone(),
                    file_id.clone(),
                    Status::Failed,
                    Some("Task failed".to_string()),
                    Some(Utc::now().to_string()),
                    &pg_pool,
                )
                .await?;
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
