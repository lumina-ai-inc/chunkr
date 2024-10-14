use crate::models::rrq::queue::QueuePayload;
use crate::models::server::extract::ExtractionPayload;
use crate::models::server::segment::Segment;
use crate::models::server::task::Status;
use crate::task::process::process_segments;
use crate::utils::db::deadpool_postgres::{create_pool, Client, Pool};
use crate::utils::json2mkd::json_2_mkd::hierarchical_chunking;
use crate::utils::storage::config_s3::create_client;
use crate::utils::storage::services::{download_to_tempfile, upload_to_s3};
use chrono::{DateTime, Utc};
use std::io::{Read, Write};
use tempfile::NamedTempFile;

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

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing task OCR");
    let s3_client = create_client().await?;
    println!("S3 client created");
    let reqwest_client = reqwest::Client::new();
    println!("Reqwest client created");
    let extraction_item: ExtractionPayload = serde_json::from_value(payload.payload)?;
    println!("Extraction payload deserialized");
    let task_id = extraction_item.task_id.clone();
    println!("Task ID: {}", task_id);
    let pg_pool = create_pool();
    println!("Postgres pool created");

    // log_task(
    //     task_id.clone(),
    //     Status::Processing,
    //     Some(format!(
    //         "Task processing | Tries ({}/{})",
    //         payload.attempt, payload.max_attempts
    //     )),
    //     None,
    //     &pg_pool,
    // )
    // .await?;

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        println!("Downloading PDF file");
        let pdf_file: NamedTempFile = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_item.input_location,
            None,
        )
        .await?;
        println!("PDF file downloaded");

        println!("Downloading output file");
        let output_file: NamedTempFile = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &extraction_item.output_location,
            None,
        )
        .await?;
        println!("Output file downloaded");

        let pdf_file_path = pdf_file.path().to_path_buf();
        println!("PDF file path: {:?}", pdf_file_path);

        println!("PDF file path: {:?}", pdf_file_path);

        println!("Reading output file contents");
        let mut file_contents = String::new();
        output_file.as_file().read_to_string(&mut file_contents)?;
        println!("Output file contents read");
        println!("Output file contents: {:?}", file_contents);

        let incomplete_segments: Vec<Segment> = match serde_json::from_str(&file_contents) {
            Ok(segments) => segments,
            Err(e) => {
                eprintln!("Error deserializing file contents: {:?}", e);
                eprintln!("File contents: {:?}", file_contents);
                return Err(Box::new(e) as Box<dyn std::error::Error>);
            }
        };
        println!("Incomplete segments deserialized");

        let processing_message = format!(
            "Processing | OCR: {}",
            extraction_item.configuration.ocr_strategy
        );
        println!("Processing message: {}", processing_message);

        println!("Logging task status");
        log_task(
            task_id.clone(),
            Status::Processing,
            Some(processing_message),
            None,
            &pg_pool,
        )
        .await?;
        println!("Task logged");
        println!("Hitting service");

        println!("Processing segments");
        let segments: Vec<Segment> =
            process_segments(&pdf_file_path, &incomplete_segments, &extraction_item).await?;
        println!("Segments processed");

        println!("Performing hierarchical chunking");
        let chunks = hierarchical_chunking(segments, extraction_item.target_chunk_length).await?;
        println!("Hierarchical chunking completed");

        println!("Writing finalized file");
        let mut finalized_file = NamedTempFile::new()?;
        finalized_file.write_all(serde_json::to_string(&chunks)?.as_bytes())?;
        println!("Finalized file created");

        println!("Uploading finalized file to S3");
        upload_to_s3(
            &s3_client,
            &extraction_item.output_location,
            &finalized_file.path(),
        )
        .await?;
        println!("Upload to S3 completed");

        println!("Cleaning up temporary files");
        if pdf_file.path().exists() {
            if let Err(e) = std::fs::remove_file(pdf_file.path()) {
                eprintln!("Error deleting temporary PDF file: {:?}", e);
            }
        }
        if output_file.path().exists() {
            if let Err(e) = std::fs::remove_file(output_file.path()) {
                eprintln!("Error deleting temporary output file: {:?}", e);
            }
        }
        if finalized_file.path().exists() {
            if let Err(e) = std::fs::remove_file(finalized_file.path()) {
                eprintln!("Error deleting temporary finalized file: {:?}", e);
            }
        }

        println!("OCR processing completed successfully");
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
            let error_message = if e
                .to_string()
                .to_lowercase()
                .contains("usage limit exceeded")
            {
                "Task failed: Usage limit exceeded".to_string()
            } else {
                "Task failed".to_string()
            };

            if payload.attempt >= payload.max_attempts {
                eprintln!("Task failed after {} attempts", payload.max_attempts);
                log_task(
                    task_id.clone(),
                    Status::Failed,
                    Some(error_message),
                    Some(Utc::now()),
                    &pg_pool,
                )
                .await?;
            }
            Err(e)
        }
    }
}
