use crate::models::rrq::{
    consume::ConsumePayload,
    queue::QueuePayload,
    status::{StatusPayload, StatusResult},
};
use crate::utils::rrq::service::{complete, consume, health};
use tokio::time::Duration;

/// This function is a consumer for the rrq queue. It will consume the queue, process the item, and then complete the item.
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
    match health().await {
        Ok(_) => (),
        Err(e) => {
            println!("Error checking health: {}", e);
            return Err(e);
        }
    };

    let consumer_payload = ConsumePayload {
        consumer_id: uuid::Uuid::new_v4().to_string(),
        queue_name: queue_name.clone(),
        item_count,
        expiration_seconds: Some(expiration_seconds),
    };

    loop {
        match consume(consumer_payload.clone()).await {
            Ok(consume_payloads) => {
                if consume_payloads.is_empty() {
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

                    let mut payloads = Vec::new();
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

                    match complete(payloads.clone()).await {
                        Ok(_) => (),
                        Err(e) => {
                            println!("Failed to complete queue item: {}. Retrying...", e);
                            payloads.iter_mut().for_each(|p| {
                                p.result = StatusResult::Success;
                            });
                            match complete(payloads).await {
                                Ok(_) => println!("Retry successful"),
                                Err(retry_e) => println!("Retry failed: {}", retry_e),
                            }
                        }
                    }
                }
            }
            Err(e) => {
                println!("Error consuming queue item: {}", e);
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        }
    }
}
