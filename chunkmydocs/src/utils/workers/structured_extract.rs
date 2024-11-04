use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::ExtractionPayload;
use crate::models::server::segment::OutputResponse;
use crate::models::server::task::Status;
use crate::utils::configs::structured_extract::Config as StructuredExtractConfig;
use crate::utils::db::deadpool_postgres::create_pool;
use crate::utils::services::structured_extract::perform_structured_extraction;
use crate::utils::services::log::log_task;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::Utc;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use tempfile::NamedTempFile;

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing structured extraction task");

    let s3_client = create_client().await?;
    let reqwest_client = reqwest::Client::new();
    let extraction_payload: ExtractionPayload = serde_json::from_value(payload.payload)?;
    let task_id = extraction_payload.task_id.clone();
    let pg_pool = create_pool();
    let config = StructuredExtractConfig::from_env()?;
    let configuration = extraction_payload.configuration.clone();
    let json_schema = configuration.json_schema.clone();

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
        let structured_results = perform_structured_extraction(
            json_schema.ok_or("JSON schema is missing")?,
            output_response.chunks.clone(),
            config.embedding_url.clone(),
            config.llm_url.clone(),
            config.llm_key.clone(),
            config.top_k as usize,
            config.model_name.clone(),
            config.batch_size as usize,
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
