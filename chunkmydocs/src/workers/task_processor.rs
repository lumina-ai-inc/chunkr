use chrono::{ DateTime, Utc };
use chunkmydocs::models::rrq::queue::QueuePayload;
use chunkmydocs::models::server::extract::ExtractionPayload;
use chunkmydocs::models::server::segment::{ PdlaSegment, BaseSegment, Segment };
use chunkmydocs::models::server::task::Status;
use chunkmydocs::task::pdla::pdla_extraction;
use chunkmydocs::task::process::process_segments;
use chunkmydocs::task::pdf::split_pdf;
use chunkmydocs::utils::configs::extraction_config;
use chunkmydocs::utils::db::deadpool_postgres::{ create_pool, Client, Pool };
use chunkmydocs::utils::json2mkd::json_2_mkd::hierarchical_chunking;
use chunkmydocs::utils::rrq::consumer::consumer;
use chunkmydocs::utils::storage::config_s3::create_client;
use chunkmydocs::utils::storage::services::{ download_to_tempfile, upload_to_s3 };
use base64::{ engine::general_purpose::STANDARD, Engine };
use image::ImageReader;
use std::{ io::Write, path::{ Path, PathBuf } };
use tempdir::TempDir;
use tempfile::NamedTempFile;

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

        let mut split_temp_files: Vec<PathBuf> = vec![];
        let split_temp_dir = TempDir::new("split_pdf")?;

        if let Some(batch_size) = extraction_item.batch_size {
            split_temp_files = split_pdf(
                temp_file.path(),
                batch_size as usize,
                split_temp_dir.path()
            ).await?;
        } else {
            split_temp_files.push(temp_file.path().to_path_buf());
        }

        let mut combined_output: Vec<Segment> = Vec::new();
        let mut page_offset: u32 = 0;

        for temp_file in &split_temp_files {
            let pdla_response = pdla_extraction(temp_file, extraction_item.model.clone()).await?;
            let pdla_segments: Vec<PdlaSegment> = serde_json::from_str(&pdla_response)?;
            let base_segments: Vec<BaseSegment> = pdla_segments.iter().map(|segment| segment.to_base_segment()).collect();
            let mut segments: Vec<Segment> = process_segments(
                temp_file,
                &base_segments,
                &extraction_item.configuration.ocr_strategy
            ).await?;

            for item in &mut segments {
                item.page_number = item.page_number + page_offset;
                if let Some(image) = &item.image {
                    match
                        (async {
                            let image_data = STANDARD.decode(image)?;
                            let mut temp_image = NamedTempFile::new()?;
                            temp_image.write_all(&image_data)?;
                            let img = ImageReader::open(temp_image.path())?.with_guessed_format()?;
                            let format = img.format().ok_or("Unable to determine image format")?;
                            
                            let output_path = Path::new(&extraction_item.output_location);
                            let parent_dir = output_path.parent()
                                .ok_or("Unable to determine parent directory")?
                                .to_str()
                                .ok_or("Invalid parent directory path")?;
    
                            let image_path = format!(
                                "{}/images/{}.{}",
                                parent_dir,
                                item.segment_id,
                                format.extensions_str()[0]
                            );
                            upload_to_s3(&s3_client, &image_path, temp_image.path()).await?;
                            Ok::<(), Box<dyn std::error::Error>>(())
                        }).await
                    {
                        Ok(_) => {}
                        Err(e) =>
                            eprintln!(
                                "Error processing image for segment {}: {:?}",
                                item.segment_id,
                                e
                            ),
                    }
                    item.image = None;
                }
            }
            combined_output.extend(segments);
            page_offset += extraction_item.batch_size.unwrap_or(1) as u32;
        }

        let chunks = hierarchical_chunking(
            combined_output,
            extraction_item.target_chunk_length
        ).await?;

        let mut output_temp_file = NamedTempFile::new()?;
        output_temp_file.write_all(serde_json::to_string(&chunks)?.as_bytes())?;

        upload_to_s3(&s3_client, &extraction_item.output_location, &output_temp_file.path()).await?;

        if temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(temp_file.path()) {
                eprintln!("Error deleting temporary file: {:?}", e);
            }
        }
        if output_temp_file.path().exists() {
            if let Err(e) = std::fs::remove_file(output_temp_file.path()) {
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = extraction_config::Config::from_env()?;
    println!("Starting task processor");
    consumer(process, config.extraction_queue, 1, 600).await?;
    Ok(())
}
