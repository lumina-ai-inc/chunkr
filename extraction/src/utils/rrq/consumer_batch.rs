use crate::models::rrq::{
    consume::{ConsumePayload, ConsumeResponse},
    queue::QueuePayload,
    status::{StatusPayload, StatusResult},
};
use crate::utils::rrq::service::{complete, consume, health};
use tokio::time::Duration;

#[derive(Debug)]
pub enum ProcessResult {
    Success(String),
    Failure(String, String),
}

pub async fn consumer_batch<F, Fut>(
    process_fn: F,
    queue_name: String,
    item_count: i64,
    expiration_seconds: u64,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(Vec<QueuePayload>) -> Fut + Send + Sync + 'static,
    Fut:
        std::future::Future<Output = Result<Vec<ProcessResult>, Box<dyn std::error::Error>>> + Send,
{
    let result = health().await?;
    println!("Health check result: {}", result);

    let consumer_payload = ConsumePayload {
        consumer_id: uuid::Uuid::new_v4().to_string(),
        queue_name: queue_name.clone(),
        item_count,
        expiration_seconds: Some(expiration_seconds),
    };

    println!(
        "Starting consumer with id: {}",
        consumer_payload.consumer_id
    );

    loop {
        let consume_payloads: Vec<ConsumeResponse> = consume(consumer_payload.clone()).await?;
        if consume_payloads.is_empty() {
            println!("No content received, waiting for 1 second before trying again...");
            tokio::time::sleep(Duration::from_secs(1)).await;
            continue;
        }

        let status_payloads: Vec<StatusPayload> = consume_payloads
            .iter()
            .map(|consume_payload| StatusPayload {
                item_id: consume_payload.queue_item.item_id.clone(),
                item_index: consume_payload.item_index,
                consumer_id: consumer_payload.consumer_id.clone(),
                queue_name: queue_name.clone(),
                result: StatusResult::Failure,
                message: None,
            })
            .collect();

        println!("Processing batch of {} items...", consume_payloads.len());

        match process_fn(
            consume_payloads
                .iter()
                .map(|c| c.queue_item.clone())
                .collect(),
        )
        .await
        {
            Ok(results) => {
                let mut payloads = Vec::new();

                for result in results {
                    match result {
                        ProcessResult::Success(item_id) => {
                            if let Some(mut status_payload) = status_payloads
                                .iter()
                                .find(|sp| sp.item_id == item_id)
                                .cloned()
                            {
                                status_payload.result = StatusResult::Success;
                                payloads.push(status_payload);
                            }
                        }
                        ProcessResult::Failure(item_id, error_message) => {
                            if let Some(mut status_payload) = status_payloads
                                .iter()
                                .find(|sp| sp.item_id == item_id)
                                .cloned()
                            {
                                status_payload.message = Some(error_message);
                                status_payload.result = StatusResult::Failure;
                                payloads.push(status_payload);
                            }
                        }
                    }
                }

                if !payloads.is_empty() {
                    match complete(payloads).await {
                        Ok(_) => println!("Batch processed"),
                        Err(e) => println!("Error completing batch: {}", e),
                    }
                }
            }
            Err(e) => {
                println!("Error processing batch: {}", e);
                match complete(status_payloads).await {
                    Ok(_) => println!("Batch failure processed for error: {}", e),
                    Err(e) => println!("Error processing batch failure: {}", e),
                }
            }
        }
    }
}
