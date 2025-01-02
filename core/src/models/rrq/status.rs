use serde::{ Deserialize, Serialize };

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum StatusResult {
    Success,
    Failure,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StatusPayload {
    pub item_id: String,
    pub item_index: i64,
    pub consumer_id: String,
    pub queue_name: String,
    pub message: Option<String>,
    pub result: StatusResult,
}