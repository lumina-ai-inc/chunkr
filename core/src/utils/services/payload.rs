use crate::models::rrq::produce::ProducePayload;
use crate::models::chunkr::upload::ExtractionPayload;
use crate::utils::rrq::service::produce;
use std::error::Error;
use uuid::Uuid;

pub async fn produce_extraction_payloads(
    queue_name: String,
    extraction_payload: ExtractionPayload
) -> Result<(), Box<dyn Error>> {

    let produce_payload = ProducePayload {
        queue_name: queue_name.clone(),
        publish_channel: None,
        payload: serde_json::to_value(extraction_payload).unwrap(),
        max_attempts: None,
        item_id: Uuid::new_v4().to_string(),
    };

    produce(vec![produce_payload]).await?;
    Ok(())
}