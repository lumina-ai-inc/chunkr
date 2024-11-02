use crate::utils::services::structured_extract::JsonSchema;
use actix_multipart::form::json::Json as MPJson;
use actix_multipart::form::{tempfile::TempFile, text::Text, MultipartForm};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use strum_macros::{Display, EnumString};
use utoipa::{IntoParams, ToSchema};

// Start of Selection
#[derive(Debug, MultipartForm, ToSchema, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct UploadForm {
    #[param(style = Form, value_type = String, format = "binary")]
    #[schema(value_type = String, format = "binary")]
    pub file: TempFile,

    #[param(style = Form, value_type = Model)]
    #[schema(value_type = Model)]
    pub model: Text<Model>,

    #[param(style = Form, value_type = Option<i32>)]
    #[schema(value_type = Option<i32>)]
    pub target_chunk_length: Option<Text<i32>>,

    #[param(style = Form, value_type = Option<OcrStrategy>)]
    #[schema(value_type = Option<OcrStrategy>)]
    pub ocr_strategy: Option<Text<OcrStrategy>>,


    pub json_schema: Option<MPJson<JsonSchema>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct ExtractionPayload {
    pub user_id: String,
    pub model: SegmentationModel,
    pub input_location: String,
    pub pdf_location: String,
    pub output_location: String,
    pub image_folder_location: String,
    pub task_id: String,
    pub batch_size: Option<i32>,
    #[serde(with = "humantime_serde")]
    pub expiration: Option<Duration>,
    pub target_chunk_length: Option<i32>,
    pub configuration: Configuration,
    pub file_name: String,
    pub page_count: Option<i32>,
}

#[derive(
    Serialize, Deserialize, Debug, Clone, Display, EnumString, Eq, PartialEq, ToSql, FromSql,
)]
pub enum SegmentationModel {
    PdlaFast,
    Pdla,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema, ToSql, FromSql)]
pub enum Model {
    Fast,
    HighQuality,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
pub struct Configuration {
    pub model: Model,
    pub ocr_strategy: OcrStrategy,
    pub target_chunk_length: Option<i32>,
    pub json_schema: Option<JsonSchema>,
}

impl Model {
    pub fn to_internal(&self) -> SegmentationModel {
        match self {
            Model::Fast => SegmentationModel::PdlaFast,
            Model::HighQuality => SegmentationModel::Pdla,
        }
    }
}

impl SegmentationModel {
    pub fn to_external(&self) -> Model {
        match self {
            SegmentationModel::PdlaFast => Model::Fast,
            SegmentationModel::Pdla => Model::HighQuality,
        }
    }

    pub fn get_extension(&self) -> &str {
        match self {
            SegmentationModel::PdlaFast => "json",
            SegmentationModel::Pdla => "json",
        }
    }
}

#[derive(
    Debug, Serialize, Deserialize, PartialEq, Clone, ToSql, FromSql, ToSchema, Display, EnumString,
)]
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
