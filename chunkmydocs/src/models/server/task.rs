use super::extract::Configuration;
use crate::models::server::segment::OutputResponse;
use chrono::{DateTime, Utc};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct TaskResponse {
    /// The unique identifier for the task.
    pub task_id: String,
    pub status: Status,
    /// The date and time when the task was created.
    pub created_at: DateTime<Utc>,
    /// The date and time when the task was finished.
    pub finished_at: Option<DateTime<Utc>>,
    /// The date and time when the task will expire.
    pub expires_at: Option<DateTime<Utc>>,
    /// A message describing the task's status or any errors that occurred.
    pub message: String,
    pub output: Option<OutputResponse>,
    /// The presigned URL of the input file.
    pub input_file_url: Option<String>,
    /// The presigned URL of the task.
    pub task_url: Option<String>,
    pub configuration: Configuration,
    /// The name of the file.
    pub file_name: Option<String>,
    /// The number of pages in the file.
    pub page_count: Option<i32>,
    /// The presigned URL of the PDF file.
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
/// The status of the task. 'Cancelled' has not yet been implemented.

pub enum Status {
    Starting,
    Processing,
    Succeeded,
    Failed,
    Canceled,
}
