use actix_multipart::form::{tempfile::TempFile, text::Text, MultipartForm};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;
use strum_macros::{Display, EnumString};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct ExtractionPayload {
    pub model: ModelInternal,
    pub input_location: String,
    pub output_location: String,
    pub task_id: String,
    pub batch_size: Option<i32>,
    #[serde(with = "humantime_serde")]
    pub expiration: Option<Duration>,
    pub target_chunk_length: Option<i32>,
    pub table_ocr: Option<TableOcr>,
}

#[derive(
    Serialize, Deserialize, Debug, Clone, Display, EnumString, Eq, PartialEq, ToSql, FromSql,
)]
pub enum ModelInternal {
    PdlaFast,
    Pdla,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema, ToSql, FromSql)]
pub enum Model {
    Fast,
    HighQuality,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema, Copy, ToSql, FromSql)]
#[postgres(name = "table_ocr")]
pub enum TableOcr {
    HTML,
    JSON,
}
impl fmt::Display for TableOcr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableOcr::HTML => write!(f, "HTML"),
            TableOcr::JSON => write!(f, "JSON"),
        }
    }
}
#[derive(Serialize, Deserialize, Debug, Clone, ToSchema, Display, Copy, PartialEq)]
pub enum TableOcrModel {
    EasyOcr,
    Tesseract,
    Qwen,
}

#[derive(Debug, MultipartForm, ToSchema)]
pub struct UploadForm {
    #[schema(value_type = String, format = "binary")]
    pub file: TempFile,
    #[schema(value_type = Model)]
    pub model: Text<Model>,
    #[schema(value_type = Option<i32>)]
    pub target_chunk_length: Option<Text<i32>>,
    #[schema(value_type = Option<TableOcr>)]
    pub table_ocr: Option<Text<TableOcr>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
pub struct Configuration {
    pub model: Model,
    pub target_chunk_length: Option<i32>,
    pub table_ocr: Option<TableOcr>,
}

impl Model {
    pub fn to_internal(&self) -> ModelInternal {
        match self {
            Model::Fast => ModelInternal::PdlaFast,
            Model::HighQuality => ModelInternal::Pdla,
        }
    }
}

impl ModelInternal {
    pub fn to_external(&self) -> Model {
        match self {
            ModelInternal::PdlaFast => Model::Fast,
            ModelInternal::Pdla => Model::HighQuality,
        }
    }

    pub fn get_extension(&self) -> &str {
        match self {
            ModelInternal::PdlaFast => "json",
            ModelInternal::Pdla => "json",
        }
    }
}
