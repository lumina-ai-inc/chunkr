use crate::models::chunkr::output::OutputResponse;
use crate::models::chunkr::segment_processing::SegmentProcessing;
use crate::models::chunkr::structured_extraction::JsonSchema;
use crate::models::chunkr::upload::{OcrStrategy, SegmentationStrategy};

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
/// The status of the task. `Cancelled` has not yet been implemented.

pub enum Status {
    Starting,
    Processing,
    Succeeded,
    Failed,
    Canceled,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// The configuration used for the task.
pub struct Configuration {
    pub expires_in: Option<i32>,
    pub json_schema: Option<JsonSchema>,
    pub ocr_strategy: OcrStrategy,
    pub segment_processing: Option<SegmentProcessing>,
    pub segmentation_strategy: Option<SegmentationStrategy>,
    pub target_chunk_length: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated]
    pub model: Option<Model>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
#[deprecated]
pub enum Model {
    Fast,
    HighQuality,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct TaskPayload {
    pub user_id: String,
    pub file_name: String,
    pub input_location: String,
    pub pdf_location: String,
    pub output_location: String,
    pub image_folder_location: String,
    pub task_id: String,
    pub configuration: Configuration,
}
