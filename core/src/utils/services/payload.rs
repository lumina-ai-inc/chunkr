use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::task::TaskPayload;
use crate::models::rrq::produce::ProducePayload;
use crate::utils::rrq::service::produce;
use std::error::Error;
use uuid::Uuid;

pub async fn produce_extraction_payloads(
    extraction_payload: TaskPayload,
) -> Result<(), Box<dyn Error>> {
    let worker_config = WorkerConfig::from_env().unwrap();
    let produce_payload = ProducePayload {
        queue_name: worker_config.queue_task,
        publish_channel: None,
        payload: serde_json::to_value(extraction_payload).unwrap(),
        max_attempts: Some(worker_config.max_retries),
        item_id: Uuid::new_v4().to_string(),
    };

    produce(vec![produce_payload]).await?;
    Ok(())
}
