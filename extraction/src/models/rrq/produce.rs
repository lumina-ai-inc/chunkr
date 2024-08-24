use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProducePayload {
    pub queue_name: String,
    pub publish_channel: Option<String>,
    pub payload: serde_json::Value,
    pub max_attempts: Option<u32>,
    pub item_id: String,
}
