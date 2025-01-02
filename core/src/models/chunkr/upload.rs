use crate::models::chunkr::segment_processing::SegmentProcessing;
use crate::models::chunkr::structured_extraction::JsonSchema;

use actix_multipart::form::json::Json as MPJson;
use actix_multipart::form::{tempfile::TempFile, text::Text, MultipartForm};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use utoipa::{IntoParams, ToSchema};

#[derive(Debug, MultipartForm, ToSchema, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct UploadForm {
    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>)]
    /// The number of seconds until task is deleted.
    /// Expried tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<Text<i32>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = String, format = "binary")]
    /// The file to be uploaded.
    pub file: TempFile,
    #[param(style = Form, value_type = Option<JsonSchema>)]
    #[schema(value_type = Option<JsonSchema>)]
    pub json_schema: Option<MPJson<JsonSchema>>,
    #[param(style = Form, value_type = Option<OcrStrategy>)]
    #[schema(value_type = Option<OcrStrategy>, default = "Auto")]
    pub ocr_strategy: Option<Text<OcrStrategy>>,
    #[param(style = Form, value_type = Option<SegmentProcessing>)]
    #[schema(value_type = Option<SegmentProcessing>)]
    pub segment_processing: Option<MPJson<SegmentProcessing>>,
    #[param(style = Form, value_type = Option<SegmentationStrategy>)]
    #[schema(value_type = Option<SegmentationStrategy>, default = "LayoutAnalysis")]
    pub segmentation_strategy: Option<Text<SegmentationStrategy>>,
    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>)]
    /// The target chunk length to be used for chunking.
    pub target_chunk_length: Option<Text<i32>>,
}

#[derive(
    Debug, Serialize, Deserialize, PartialEq, Clone, ToSql, FromSql, ToSchema, Display, EnumString,
)]
/// Controls the Optical Character Recognition (OCR) strategy.
/// - `All`: Processes all pages with OCR.
/// - `Auto`: Selectively applies OCR only to pages with missing or low-quality text. This works for most documents and is faster.
pub enum OcrStrategy {
    All,
    #[serde(alias = "Off")]
    Auto,
}

impl Default for OcrStrategy {
    fn default() -> Self {
        OcrStrategy::Auto
    }
}

#[derive(
    Serialize,
    Deserialize,
    Debug,
    Clone,
    Display,
    EnumString,
    Eq,
    PartialEq,
    ToSql,
    FromSql,
    ToSchema,
)]
/// Controls the segmentation strategy:
/// - `LayoutAnalysis`: Analyzes pages for layout elements (e.g., `Table`, `Picture`, `Formula`, etc.) using bounding boxes. Provides fine-grained segmentation and better chunking, but with a latency penalty.
/// - `Page`: Treats each page as a single segment. Faster processing, but without layout element detection and only simple chunking.
pub enum SegmentationStrategy {
    LayoutAnalysis,
    Page,
}

impl Default for SegmentationStrategy {
    fn default() -> Self {
        SegmentationStrategy::LayoutAnalysis
    }
}
