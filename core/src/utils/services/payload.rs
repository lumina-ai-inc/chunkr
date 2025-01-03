use crate::models::chunkr::task::TaskPayload;
use crate::models::rrq::produce::ProducePayload;
use crate::utils::rrq::service::produce;
use crate::utils::configs::worker_config::Config as WorkerConfig;
use std::error::Error;
use uuid::Uuid;

pub async fn produce_extraction_payloads(
    queue_name: String,
    extraction_payload: TaskPayload,
) -> Result<(), Box<dyn Error>> {
    let worker_config = WorkerConfig::from_env().unwrap();
    let produce_payload = ProducePayload {
        queue_name: queue_name.clone(),
        publish_channel: None,
        payload: serde_json::to_value(extraction_payload).unwrap(),
        max_attempts: Some(worker_config.max_retries),
        item_id: Uuid::new_v4().to_string(),
    };

    produce(vec![produce_payload]).await?;
    Ok(())
}
