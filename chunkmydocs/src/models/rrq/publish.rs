use serde::{ Deserialize, Serialize };

#[derive(Debug, Deserialize, Serialize)]
pub struct PublishPayload {
    pub item_id: String,
    pub consumer_id: String,
    pub queue_name: String,
    pub payload: serde_json::Value,
    pub success: bool,
    pub message: Option<String>
}