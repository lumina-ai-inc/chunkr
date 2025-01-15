use crate::models::chunkr::chunk_processing::ChunkProcessing;
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
pub struct CreateForm {
    #[param(style = Form, value_type = Option<ChunkProcessing>)]
    #[schema(value_type = Option<ChunkProcessing>)]
    pub chunk_processing: Option<MPJson<ChunkProcessing>>,
    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>)]
    /// The number of seconds until task is deleted.
    /// Expried tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<MPJson<i32>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = String, format = "binary")]
    /// The file to be uploaded.
    pub file: TempFile,
    #[param(style = Form, value_type = Option<bool>)]
    #[schema(value_type = Option<bool>, default = false)]
    /// Whether to use high-resolution images for cropping and post-processing. (Latency penalty: ~7 seconds per page)
    pub high_resolution: Option<MPJson<bool>>,
    #[param(style = Form, value_type = Option<JsonSchema>)]
    #[schema(value_type = Option<JsonSchema>)]
    pub json_schema: Option<MPJson<JsonSchema>>,
    #[param(style = Form, value_type = Option<OcrStrategy>)]
    #[schema(value_type = Option<OcrStrategy>, default = "All")]
    pub ocr_strategy: Option<MPJson<OcrStrategy>>,
    #[param(style = Form, value_type = Option<SegmentProcessing>)]
    #[schema(value_type = Option<SegmentProcessing>)]
    pub segment_processing: Option<MPJson<SegmentProcessing>>,
    #[param(style = Form, value_type = Option<SegmentationStrategy>)]
    #[schema(value_type = Option<SegmentationStrategy>, default = "LayoutAnalysis")]
    pub segmentation_strategy: Option<MPJson<SegmentationStrategy>>,
    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>, default = 512)]
    #[deprecated = "Use `chunk_processing` instead"]
    /// Deprecated: Use `chunk_processing.target_length` instead.
    ///
    /// The target chunk length to be used for chunking.
    /// If 0, each chunk will contain a single segment.
    pub target_chunk_length: Option<Text<i32>>,
}

impl CreateForm {
    pub fn get_chunk_processing(&self) -> Option<ChunkProcessing> {
        self.chunk_processing
            .as_ref()
            .map(|mp_json| mp_json.0.clone())
            .or_else(|| {
                // For backwards compatibility: if chunk_processing is not set but target_chunk_length is,
                // create a ChunkProcessing with defaults and override target_length
                self.target_chunk_length.as_ref().map(|length| {
                    let mut chunk_processing = ChunkProcessing::default();
                    chunk_processing.target_length = length.0;
                    chunk_processing
                })
            })
    }
}

#[derive(Debug, MultipartForm, ToSchema, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct UpdateForm {
    #[param(style = Form, value_type = Option<ChunkProcessing>)]
    #[schema(value_type = Option<ChunkProcessing>)]
    pub chunk_processing: Option<MPJson<ChunkProcessing>>,
    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>)]
    /// The number of seconds until task is deleted.
    /// Expried tasks can **not** be updated, polled or accessed via web interface.
    pub expires_in: Option<MPJson<i32>>,
    #[param(style = Form, value_type = Option<bool>)]
    #[schema(value_type = Option<bool>)]
    /// Whether to use high-resolution images for cropping and post-processing. (Latency penalty: ~7 seconds per page)
    pub high_resolution: Option<MPJson<bool>>,
    #[param(style = Form, value_type = Option<JsonSchema>)]
    #[schema(value_type = Option<JsonSchema>)]
    pub json_schema: Option<MPJson<JsonSchema>>,
    #[param(style = Form, value_type = Option<OcrStrategy>)]
    #[schema(value_type = Option<OcrStrategy>)]
    pub ocr_strategy: Option<MPJson<OcrStrategy>>,
    #[param(style = Form, value_type = Option<SegmentProcessing>)]
    #[schema(value_type = Option<SegmentProcessing>)]
    pub segment_processing: Option<MPJson<SegmentProcessing>>,
    #[param(style = Form, value_type = Option<SegmentationStrategy>)]
    #[schema(value_type = Option<SegmentationStrategy>)]
    pub segmentation_strategy: Option<MPJson<SegmentationStrategy>>,
}

impl UpdateForm {
    pub fn get_chunk_processing(&self) -> Option<ChunkProcessing> {
        self.chunk_processing
            .as_ref()
            .map(|mp_json| mp_json.0.clone())
    }
}

#[derive(
    Debug, Serialize, Deserialize, PartialEq, Clone, ToSql, FromSql, ToSchema, Display, EnumString,
)]
/// Controls the Optical Character Recognition (OCR) strategy.
/// - `All`: Processes all pages with OCR. (Latency penalty: ~0.5 seconds per page)
/// - `Auto`: Selectively applies OCR only to pages with missing or low-quality text. When text layer is present the bounding boxes from the text layer are used.
pub enum OcrStrategy {
    All,
    #[serde(alias = "Off")]
    Auto,
}

impl Default for OcrStrategy {
    fn default() -> Self {
        OcrStrategy::All
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
/// - `LayoutAnalysis`: Analyzes pages for layout elements (e.g., `Table`, `Picture`, `Formula`, etc.) using bounding boxes. Provides fine-grained segmentation and better chunking. (Latency penalty: ~TBD seconds per page).
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
