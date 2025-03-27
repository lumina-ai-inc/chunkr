use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct QueuePayload {
    pub queue_name: String,
    pub publish_channel: Option<String>,
    pub attempt: u32,
    pub max_attempts: u32,
    pub payload: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub item_id: String,
}
