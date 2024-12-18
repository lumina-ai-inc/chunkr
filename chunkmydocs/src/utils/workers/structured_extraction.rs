use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::ExtractionPayload;
use crate::models::server::segment::OutputResponse;
use crate::models::server::task::Status;
use crate::utils::configs::llm_config::Config as LlmConfig;
use crate::utils::configs::s3_config::create_client;
use crate::utils::configs::search_config::Config as SearchConfig;
use crate::utils::configs::worker_config::Config as WorkerConfig;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::services::{log::log_task, structured_extraction::perform_structured_extraction};
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::Utc;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use tempfile::NamedTempFile;

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task");
    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_payload.task_id.clone();
    let pg_pool = create_pool();
    let worker_config = WorkerConfig::from_env().expect("Failed to load WorkerConfig");
    let search_config = SearchConfig::from_env().expect("Failed to load SearchConfig");
    let llm_config = LlmConfig::from_env().expect("Failed to load LlmConfig");
    let configuration = extraction_payload.configuration.clone();
    let json_schema = configuration.json_schema.clone();
    let content_type = "markdown";

    let result: Result<(), Box<dyn std::error::Error>> = async {
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Structured extraction started".to_string()),
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
        println!("starting structured extraction");
        let structured_results = perform_structured_extraction(
            json_schema.ok_or("JSON schema is missing")?,
            output_response.chunks.clone(),
            format!("{}/embed", search_config.dense_vector_url),
            llm_config
                .structured_extraction_url
                .clone()
                .unwrap_or(llm_config.url.clone()),
            llm_config
                .structured_extraction_key
                .clone()
                .unwrap_or(llm_config.key.clone()),
            worker_config.structured_extraction_top_k as usize,
            llm_config
                .structured_extraction_model
                .clone()
                .unwrap_or(llm_config.model.clone()),
            worker_config.structured_extraction_batch_size as usize,
            content_type.to_string(),
        )
        .await
        .map_err(|e| e.to_string())?;
        output_response.extracted_json = Some(structured_results);

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&output_response)?.as_bytes())?;

        upload_to_s3(
            &s3_client,
            &extraction_payload.output_location,
            &output_temp_file.path(),
        )
        .await
        .map_err(|e| e.to_string())?;

        if output_temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(output_temp_file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
            }
        }

        Ok(())
    }
    .await;

    match result {
        Ok(_) => {
            println!("Structured extraction task succeeded");
            log_task(
                task_id.clone(),
                Status::Succeeded,
                Some("Structured extraction succeeded".to_string()),
                Some(Utc::now()),
                &pg_pool,
            )
            .await?;

            Ok(())
        }
        Err(e) => {
            eprintln!("Error processing structured extraction task: {:?}", e);
            if payload.attempt >= payload.max_attempts {
                eprintln!("Task failed after {} attempts", payload.max_attempts);

                log_task(
                    task_id.clone(),
                    Status::Failed,
                    Some("Structured extraction failed".to_string()),
                    Some(Utc::now()),
                    &pg_pool,
                )
                .await?;
            }
            Err(e)
        }
    }
}
