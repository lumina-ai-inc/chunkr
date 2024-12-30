use crate::utils::services::structured_extraction::JsonSchema;
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
    /// Not recommended - as expried tasks can not be updated, polled or accessed via web interface.
    pub expires_in: Option<Text<i32>>,
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = String, format = "binary")]
    /// The file to be uploaded.
    pub file: TempFile,
    #[param(style = Form, value_type = Option<JsonSchema>)]
    #[schema(value_type = Option<JsonSchema>)]
    /// The JSON schema to be used for structured extraction.
    pub json_schema: Option<MPJson<JsonSchema>>,
    #[param(style = Form, value_type = Option<Model>)]
    #[schema(value_type = Option<Model>, default = "HighQuality")]
    pub model: Option<Text<Model>>,
    #[param(style = Form, value_type = Option<OcrStrategy>)]
    #[schema(value_type = Option<OcrStrategy>, default = "Auto")]
    pub ocr_strategy: Option<Text<OcrStrategy>>,
    #[param(style = Form, value_type = Option<SegmentationStrategy>)]
    #[schema(value_type = Option<SegmentationStrategy>, default = "LayoutAnalysis")]
    pub segmentation_strategy: Option<Text<SegmentationStrategy>>,
    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>)]
    /// The target chunk length to be used for chunking.
    pub target_chunk_length: Option<Text<i32>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct TaskPayload {
    pub user_id: String,
    pub model: PdlaModel,
    pub input_location: String,
    pub pdf_location: String,
    pub output_location: String,
    pub image_folder_location: String,
    pub task_id: String,
    pub batch_size: Option<i32>,
    pub target_chunk_length: Option<i32>,
    pub configuration: Configuration,
    pub file_name: String,
    pub page_count: Option<i32>,
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
/// The segmentation strategy to be used.
/// If set to 'LayoutAnalysis', the document will be analyzed for layout elements (like paragraphs, tables, headers) using bounding boxes for more granular segmentation.
/// If set to 'Page', the document will be segmented by pages only, treating each page as a single segment.
pub enum SegmentationStrategy {
    LayoutAnalysis,
    Page,
}

#[derive(
    Serialize, Deserialize, Debug, Clone, Display, EnumString, Eq, PartialEq, ToSql, FromSql,
)]
pub enum PdlaModel {
    PdlaFast,
    Pdla,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema, ToSql, FromSql)]
/// The model to be used for segmentation.
pub enum Model {
    Fast,
    HighQuality,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
/// The configuration used for the task.
pub struct Configuration {
    pub model: Model,
    pub ocr_strategy: OcrStrategy,
    pub target_chunk_length: Option<i32>,
    pub json_schema: Option<JsonSchema>,
    pub segmentation_strategy: Option<SegmentationStrategy>,
    pub expires_in: Option<i32>,
}

impl Model {
    pub fn to_internal(&self) -> PdlaModel {
        match self {
            Model::Fast => PdlaModel::PdlaFast,
            Model::HighQuality => PdlaModel::Pdla,
        }
    }
}

impl PdlaModel {
    pub fn to_external(&self) -> Model {
        match self {
            PdlaModel::PdlaFast => Model::Fast,
            PdlaModel::Pdla => Model::HighQuality,
        }
    }

    pub fn get_extension(&self) -> &str {
        match self {
            PdlaModel::PdlaFast => "json",
            PdlaModel::Pdla => "json",
        }
    }
}

#[derive(
    Debug, Serialize, Deserialize, PartialEq, Clone, ToSql, FromSql, ToSchema, Display, EnumString,
)]
/// The OCR strategy to be used.
pub enum OcrStrategy {
    Auto,
    All,
    Off,
}

impl Default for OcrStrategy {
    fn default() -> Self {
        OcrStrategy::Auto
    }
}
