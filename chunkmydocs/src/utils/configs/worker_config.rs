use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum GeneralOcrModel {
    Doctr,
    Paddle,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TableOcrModel {
    LLM,
    Paddle,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum PageOcrModel {
    LLM,
    Doctr,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_batch_size")]
    pub batch_size: i32,
    #[serde(default = "default_general_ocr_model")]
    pub general_ocr_model: GeneralOcrModel,
    #[serde(default = "default_general_ocr_url")]
    pub general_ocr_url: Option<String>,
    #[serde(default = "default_ocr_confidence_threshold")]
    pub ocr_confidence_threshold: f32,
    #[serde(default = "default_page_image_density")]
    pub page_image_density: f32,
    #[serde(default = "default_page_limit")]
    pub page_limit: i32,
    #[serde(default = "default_pdla_fast_url")]
    pub pdla_fast_url: String,
    #[serde(default = "default_pdla_url")]
    pub pdla_url: String,
    #[serde(default = "default_pdf_density")]
    pub pdf_density: f32,
    #[serde(default = "default_queue_fast")]
    pub queue_fast: String,
    #[serde(default = "default_queue_high_quality")]
    pub queue_high_quality: String,
    #[serde(default = "default_queue_ocr")]
    pub queue_ocr: String,
    #[serde(default = "default_queue_postprocess")]
    pub queue_postprocess: String,
    #[serde(default = "default_queue_preprocess")]
    pub queue_preprocess: String,
    #[serde(default = "default_queue_structured_extraction")]
    pub queue_structured_extraction: String,
    #[serde(default = "default_s3_bucket")]
    pub s3_bucket: String,
    #[serde(default = "default_segment_bbox_offset")]
    pub segment_bbox_offset: f32,
    #[serde(default = "default_server_url")]
    pub server_url: String,
    #[serde(default = "default_structured_extraction_batch_size")]
    pub structured_extraction_batch_size: i32,
    #[serde(default = "default_structured_extraction_top_k")]
    pub structured_extraction_top_k: i32,
    #[serde(default = "default_table_ocr_model")]
    pub table_ocr_model: TableOcrModel,
    #[serde(default = "default_table_ocr_url")]
    pub table_ocr_url: Option<String>,
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_page_ocr_model")]
    pub page_ocr_model: PageOcrModel,
}

fn default_page_ocr_model() -> PageOcrModel {
    PageOcrModel::LLM
}

fn default_batch_size() -> i32 {
    300
}

fn default_general_ocr_model() -> GeneralOcrModel {
    GeneralOcrModel::Doctr
}

fn default_general_ocr_url() -> Option<String> {
    Some("http://localhost:8003".to_string())
}

fn default_ocr_confidence_threshold() -> f32 {
    0.85
}

fn default_page_image_density() -> f32 {
    150.0
}

fn default_page_limit() -> i32 {
    500
}

fn default_pdla_fast_url() -> String {
    "http://localhost:8002".to_string()
}

fn default_pdla_url() -> String {
    "http://localhost:8002".to_string()
}

fn default_pdf_density() -> f32 {
    72.0
}

fn default_queue_fast() -> String {
    "fast".to_string()
}

fn default_queue_high_quality() -> String {
    "high-quality".to_string()
}

fn default_queue_ocr() -> String {
    "ocr".to_string()
}

fn default_queue_postprocess() -> String {
    "postprocess".to_string()
}

fn default_queue_preprocess() -> String {
    "preprocess".to_string()
}

fn default_queue_structured_extraction() -> String {
    "structured-extraction".to_string()
}

fn default_s3_bucket() -> String {
    "chunkr".to_string()
}

fn default_segment_bbox_offset() -> f32 {
    1.0
}

fn default_server_url() -> String {
    "http://localhost:8000".to_string()
}

fn default_structured_extraction_batch_size() -> i32 {
    32
}

fn default_structured_extraction_top_k() -> i32 {
    45
}

fn default_table_ocr_model() -> TableOcrModel {
    TableOcrModel::Paddle
}

fn default_table_ocr_url() -> Option<String> {
    Some("http://localhost:8004".to_string())
}

fn default_version() -> String {
    "1.0.3".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("WORKER")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
