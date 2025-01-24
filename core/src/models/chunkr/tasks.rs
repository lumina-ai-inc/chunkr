use serde::Deserialize;

#[derive(Deserialize)]
pub struct TasksQuery {
    pub page: Option<i64>,
    pub limit: Option<i64>,
    pub include_chunks: Option<bool>,
    pub start: Option<chrono::DateTime<chrono::Utc>>,
    pub end: Option<chrono::DateTime<chrono::Utc>>,
}
