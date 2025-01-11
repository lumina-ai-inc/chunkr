use crate::models::chunkr::chunk_processing::ChunkProcessing;
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
    pub configuration: Configuration,
    /// The date and time when the task was created and queued.
    pub created_at: DateTime<Utc>,
    /// The date and time when the task will expire.
    pub expires_at: Option<DateTime<Utc>>,
    /// The name of the file.
    pub file_name: Option<String>,
    /// The date and time when the task was finished.
    pub finished_at: Option<DateTime<Utc>>,
    /// The presigned URL of the input file.
    pub input_file_url: Option<String>,
    /// A message describing the task's status or any errors that occurred.
    pub message: String,
    pub output: Option<OutputResponse>,
    /// The number of pages in the file.
    pub page_count: Option<i32>,
    /// The presigned URL of the PDF file.
    pub pdf_url: Option<String>,
    /// The date and time when the task was started.
    pub started_at: Option<DateTime<Utc>>,
    pub status: Status,
    /// The unique identifier for the task.
    pub task_id: String,
    /// The presigned URL of the task.
    pub task_url: Option<String>,
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
/// The status of the task.

pub enum Status {
    Starting,
    Processing,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// The configuration used for the task.
pub struct Configuration {
    pub chunk_processing: ChunkProcessing,
    #[serde(alias = "expires_at")]
    /// The number of seconds until task is deleted.
    /// Expried tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<i32>,
    /// Whether to use high-resolution images for cropping and post-processing.
    pub high_resolution: bool,
    pub json_schema: Option<JsonSchema>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated]
    pub model: Option<Model>,
    pub ocr_strategy: OcrStrategy,
    pub segment_processing: SegmentProcessing,
    pub segmentation_strategy: SegmentationStrategy,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated]
    /// The target number of words in each chunk. If 0, each chunk will contain a single segment.
    pub target_chunk_length: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, ToSql, FromSql)]
#[deprecated]
pub enum Model {
    Fast,
    HighQuality,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct TaskPayload {
    pub current_configuration: Configuration,
    pub file_name: String,
    pub image_folder_location: String,
    pub input_location: String,
    pub output_location: String,
    pub pdf_location: String,
    pub previous_configuration: Option<Configuration>,
    pub task_id: String,
    pub user_id: String,
}
