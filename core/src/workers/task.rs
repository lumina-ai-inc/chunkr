use chrono::Utc;
use core::models::chunkr::pipeline::Pipeline;
use core::models::chunkr::task::Status;
use core::models::chunkr::task::TaskPayload;
use core::models::rrq::queue::QueuePayload;
use core::pipeline::convert_to_images;
use core::pipeline::pages;
use core::pipeline::update_metadata;
use core::configs::pdfium_config::Config as PdfiumConfig;
use core::configs::s3_config::create_client;
use core::configs::worker_config::Config as WorkerConfig;
use core::configs::postgres_config::{create_pool, Pool};
use core::utils::rrq::consumer::consumer;
use core::utils::services::log::log_task;
use core::utils::storage::services::download_to_tempfile;

async fn execute_step(
    step: &str,
    pipeline: &mut Pipeline,
    pg_pool: &Pool,
) -> Result<(Status, Option<String>), Box<dyn std::error::Error>> {
    println!("Executing step: {}", step);
    let start = std::time::Instant::now();
    let result = match step {
        "convert_to_images" => convert_to_images::process(pipeline).await,
        "pages" => pages::process(pipeline).await,
        "update_metadata" => update_metadata::process(pipeline, pg_pool).await,
        _ => Err(format!("Unknown function: {}", step).into()),
    }?;
    let duration = start.elapsed();
    println!(
        "Step {} took {:?} with page count {:?}",
        step,
        duration,
        pipeline.page_count.unwrap_or(0)
    );
    log_task(
        &pipeline.task_payload.task_id,
        result.0.clone(),
        Some(&result.1.clone().unwrap_or_default()),
        None,
        &pg_pool,
    )
    .await?;
    Ok(result)
}

fn orchestrate_task() -> Vec<&'static str> {
    vec!["update_metadata", "convert_to_images", "pages"]
}

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    let pg_pool = create_pool();
    let reqwest_client = reqwest::Client::new();
    let s3_client = create_client().await?;
    let task_payload: TaskPayload = serde_json::from_value(payload.payload)?;
    let task_id = task_payload.task_id.clone();

    let result: Result<(), Box<dyn std::error::Error>> = (async {
        log_task(
            &task_id,
            Status::Processing,
            Some("Task started"),
            None,
            &pg_pool,
        )
        .await?;

        let input_file = download_to_tempfile(
            &s3_client,
            &reqwest_client,
            &task_payload.input_location,
            None,
        )
        .await
        .map_err(|e| {
            println!("Failed to download input file: {:?}", e);
            e
        })?;

        let mut pipeline = match Pipeline::new(input_file, task_payload) {
            Ok(pipeline) => pipeline,
            Err(e) => {
                if e.to_string().contains("Unsupported file type") {
                    log_task(
                        &task_id,
                        Status::Failed,
                        Some(&e.to_string()),
                        Some(Utc::now()),
                        &pg_pool,
                    )
                    .await?;
                    return Ok(());
                }
                return Err(e.into());
            }
        };

        for step in orchestrate_task() {
            let result: (Status, Option<String>) =
                execute_step(step, &mut pipeline, &pg_pool).await?;
            if result.0 == Status::Failed {
                return Ok(());
            }
        }

        // TODO: Change status to succeeded after development
        log_task(
            &task_id,
            Status::Failed,
            Some(&"Task succeeded".to_string()),
            Some(Utc::now()),
            &pg_pool,
        )
        .await?;

        Ok(())
    })
    .await;

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            let message = match payload.attempt >= payload.max_attempts {
                true => "Task failed".to_string(),
                false => format!("Retrying task {}/{}", payload.attempt, payload.max_attempts),
            };
            log_task(
                &task_id,
                Status::Failed,
                Some(&message),
                Some(Utc::now()),
                &pg_pool,
            )
            .await?;
            Err(e)
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting task processor");
    let config = WorkerConfig::from_env()?;
    PdfiumConfig::from_env()?.ensure_binary().await?;
    consumer(process, config.queue_task, 1, 2400).await?;
    Ok(())
}
