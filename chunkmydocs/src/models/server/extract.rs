use actix_multipart::form::{ tempfile::TempFile, text::Text, MultipartForm };
use serde::{ Deserialize, Serialize };
use std::time::Duration;
use strum_macros::{ Display, EnumString };
use utoipa::ToSchema;
use postgres_types::{ FromSql, ToSql };

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub struct ExtractionPayload {
    pub model: ModelInternal,
    pub input_location: String,
    pub output_location: String,
    pub task_id: String,
    pub file_id: String,
    pub batch_size: Option<i32>,
    #[serde(with = "humantime_serde")]
    pub expiration: Option<Duration>,
    pub target_chunk_length: Option<i32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Display, EnumString, Eq, PartialEq, ToSql, FromSql)]
pub enum ModelInternal {
    Grobid,
    PdlaFast,
    Pdla,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema, ToSql, FromSql)]
pub enum Model {
    Research,
    Fast,
    HighQuality,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub enum TableOcr {
    HTML,
    JSON,
}

#[derive(Serialize, Deserialize, Debug, Clone, ToSchema)]
pub enum TableOcrModel {
    EasyOcr,
    Tesseract,
}

#[derive(Debug, MultipartForm, ToSchema)]
pub struct UploadForm {
    pub file: TempFile,
    pub model: Text<Model>,
    pub target_chunk_length: Option<Text<i32>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
pub struct Configuration {
    pub model: Model,
    pub target_chunk_length: Option<i32>,
}

impl Model {
    pub fn to_internal(&self) -> ModelInternal {
        match self {
            Model::Research => ModelInternal::Grobid,
            Model::Fast => ModelInternal::PdlaFast,
            Model::HighQuality => ModelInternal::Pdla,
        }
    }
}

impl ModelInternal {
    pub fn to_external(&self) -> Model {
        match self {
            ModelInternal::Grobid => Model::Research,
            ModelInternal::PdlaFast => Model::Fast,
            ModelInternal::Pdla => Model::HighQuality,
        }
    }

    pub fn get_extension(&self) -> &str {
        match self {
            ModelInternal::Grobid => "xml",
            ModelInternal::PdlaFast => "json",
            ModelInternal::Pdla => "json",
        }
    }
}
