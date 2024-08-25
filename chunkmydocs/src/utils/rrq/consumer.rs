use crate::models::rrq::{
    consume::{ConsumePayload, ConsumeResponse},
    queue::QueuePayload,
    status::{StatusPayload, StatusResult},
};
use crate::utils::rrq::service::{complete, consume, health};
use tokio::time::Duration;

pub async fn consumer<F, Fut>(
    process_fn: F,
    queue_name: String,
    item_count: i64,
    expiration_seconds: u64,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(QueuePayload) -> Fut + Sync + 'static,
    Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error>>>,
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

        for consume_payload in consume_payloads {
            let mut status_payload = StatusPayload {
                item_id: consume_payload.queue_item.item_id.clone(),
                item_index: consume_payload.item_index,
                consumer_id: consumer_payload.consumer_id.clone(),
                queue_name: queue_name.clone(),
                result: StatusResult::Failure,
                message: None,
            };

            println!(
                "Processing queue item: {}",
                consume_payload.queue_item.item_id
            );

            let mut payloads = Vec::new();
            println!("Processing queue item2");
            match process_fn(consume_payload.queue_item).await {
                Ok(_) => {
                    status_payload.result = StatusResult::Success;
                }
                Err(e) => {
                    println!("Error processing queue item: {}", e);
                    status_payload.message = Some(e.to_string());
                    status_payload.result = StatusResult::Failure;
                }
            }
            payloads.push(status_payload);

            println!("Payloads: {:?}", payloads);

            match complete(payloads.clone()).await {
                Ok(_) => (),
                Err(e) => {
                    println!("Failed to complete queue item: {}. Retrying...", e);
                    payloads.iter_mut().for_each(|p| {
                        p.result = StatusResult::Success;
                    });
                    // Retry once
                    match complete(payloads).await {
                        Ok(_) => println!("Retry successful"),
                        Err(retry_e) => println!("Retry failed: {}", retry_e),
                    }
                }
            }
        }
    }
}
