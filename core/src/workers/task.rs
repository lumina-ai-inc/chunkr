use chrono::Utc;
use core::models::chunkr::task::Status;
use core::models::chunkr::task::TaskPayload;
use core::models::rrq::queue::QueuePayload;
use core::utils::configs::s3_config::create_client;
use core::utils::configs::worker_config::Config as WorkerConfig;
use core::utils::db::deadpool_postgres::{create_pool, Client};
use core::utils::rrq::consumer::consumer;
use core::utils::services::log::log_task;
use core::utils::storage::services::download_to_given_tempfile;
use serde_json::Value;
use tempfile::{NamedTempFile, TempDir};

// TODO: save outputs from each step so that future steps can use them
// TODO: or save the input for a step in a Hashmap so that future steps can use them
async fn execute_step(
    step: String,
    config: &mut Value,
    s3_client: &aws_sdk_s3::Client,
    reqwest_client: &reqwest::Client,
    temp_dir: &TempDir,
) -> Result<Value, Box<dyn std::error::Error>> {
    match step {
        // "function1" => execute_function1(config).await,
        // "function2" => execute_function2(config).await,
        _ => Err(format!("Unknown function: {}", step).into()),
    }
}

fn orchestrate_task(task_payload: TaskPayload, client: &Client) -> Vec<String> {
    unimplemented!()
}

pub async fn process(payload: QueuePayload) -> Result<(), Box<dyn std::error::Error>> {
    let pg_pool = create_pool();
    let client: Client = pg_pool.get().await?;
    let reqwest_client = reqwest::Client::new();
    let s3_client: aws_sdk_s3::Client = create_client().await?;
    let task_payload: TaskPayload = serde_json::from_value(payload.payload)?;
    let task_id = task_payload.task_id.clone();
    let temp_dir = TempDir::new().unwrap();

    let result: Result<(Status, Option<String>), Box<dyn std::error::Error>> = (async {
        log_task(
            task_id.clone(),
            Status::Processing,
            Some("Task started".to_string()),
            None,
            &pg_pool,
        )
        .await?;

        let mut input_file = NamedTempFile::new_in(&temp_dir)?;
        download_to_given_tempfile(
            &mut input_file,
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

        let steps = orchestrate_task(task_payload, &client);

        let mut config: Value = Value::Null;
        for step in steps {
            let _ = execute_step(step, &mut config, &s3_client, &reqwest_client, &temp_dir).await?;
        }

        Ok((Status::Succeeded, Some("Task succeeded".to_string())))
    })
    .await;

    match result {
        Ok(value) => {
            log_task(
                task_id.clone(),
                value.0,
                value.1,
                Some(Utc::now()),
                &pg_pool,
            )
            .await?;
            Ok(())
        }
        Err(e) => {
            let message = match payload.attempt >= payload.max_attempts {
                true => "Task failed".to_string(),
                false => format!("Retrying task {}/{}", payload.attempt, payload.max_attempts),
            };
            log_task(
                task_id.clone(),
                Status::Failed,
                Some(message),
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
    consumer(process, config.queue_task, 1, 2400).await?;
    Ok(())
}
