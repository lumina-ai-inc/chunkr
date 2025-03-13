use super::queue::QueuePayload;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ConsumePayload {
    pub consumer_id: String,
    pub queue_name: String,
    pub item_count: i64,
    pub expiration_seconds: Option<u64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ConsumeResponse {
    pub queue_item: QueuePayload,
    pub item_index: i64,
    pub consumed_at: chrono::DateTime<chrono::Utc>,
}
