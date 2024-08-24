use actix_multipart::form::{ tempfile::TempFile, MultipartForm, text::Text };
use serde::{ Serialize, Deserialize };
use std::time::Duration;
use strum_macros::{ Display, EnumString };

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExtractionPayload {
    pub model: ModelInternal,
    pub input_location: String,
    pub output_location: String,
    pub task_id: String,
    pub file_id: String,
    pub batch_size: Option<i32>,
    #[serde(with = "humantime_serde")]
    pub expiration: Option<Duration>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Display, EnumString, Eq, PartialEq)]
pub enum ModelInternal {
    Grobid,
    PdlaFast,
    Pdla,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Model {
    Research,
    Fast,
    HighQuality,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TableOcr {
    HTML,
    JSON,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TableOcrModel {
    EasyOcr,
    Tesseract,
}

#[derive(Debug, MultipartForm)]
pub struct UploadForm {
    pub file: TempFile,
    pub model: Text<Model>,
    pub table_ocr: Option<Text<TableOcr>>,
    pub table_ocr_model: Option<Text<TableOcrModel>>,
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

