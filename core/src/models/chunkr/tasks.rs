use serde::Deserialize;

#[derive(Deserialize)]
pub struct TasksQuery {
    pub base64_urls: Option<bool>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
    pub include_chunks: Option<bool>,
    pub limit: Option<i64>,
    pub page: Option<i64>,
    pub start: Option<chrono::DateTime<chrono::Utc>>,
}
