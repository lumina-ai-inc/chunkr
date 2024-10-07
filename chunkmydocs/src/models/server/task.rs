use super::extract::Configuration;
use crate::models::server::segment::Chunk;
use chrono::{DateTime, Utc};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct TaskResponse {
    pub task_id: String,
    pub status: Status,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub message: String,
    pub output: Option<Vec<Chunk>>,
    pub input_file_url: Option<String>,
    pub task_url: Option<String>,
    pub configuration: Configuration,
    pub file_name: Option<String>,
    pub page_count: Option<i32>,
    pub pdf_url: Option<String>,
}

#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    ToSql,
    FromSql,
    PartialEq,
    Eq,
    EnumString,
    Display,
    ToSchema,
)]
pub enum Status {
    Starting,
    Processing,
    Succeeded,
    Failed,
    Canceled,
}
