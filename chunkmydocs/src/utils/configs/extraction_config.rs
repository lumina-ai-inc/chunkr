use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_version")]
    pub version: String,
    #[serde(default = "default_queue_preprocess")]
    pub queue_preprocess: String,
    #[serde(default = "default_queue_fast")]
    pub queue_fast: String,
    #[serde(default = "default_queue_high_quality")]
    pub queue_high_quality: String,
    #[serde(default = "default_queue_postprocess")]
    pub queue_postprocess: String,
    #[serde(default = "default_queue_ocr")]
    pub queue_ocr: String,
    #[serde(default = "default_queue_structured_extract")]
    pub queue_structured_extract: String,
    #[serde(default = "default_pdla_url")]
    pub pdla_url: String,
    #[serde(default = "default_pdla_fast_url")]
    pub pdla_fast_url: String,
    #[serde(default = "default_general_ocr_url")]
    pub general_ocr_url: Option<String>,
    #[serde(default = "default_table_ocr_url")]
    pub table_ocr_url: Option<String>,
    #[serde(with = "duration_seconds", default = "default_task_expiration")]
    pub task_expiration: Option<Duration>,
    #[serde(default = "default_s3_bucket")]
    pub s3_bucket: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: i32,
    #[serde(default = "default_server_url")]
    pub server_url: String,
    #[serde(default = "default_ocr_concurrency")]
    pub ocr_concurrency: usize,
    #[serde(default = "default_ocr_confidence_threshold")]
    pub ocr_confidence_threshold: f32,
    #[serde(default = "default_pdf_density")]
    pub pdf_density: f32,
    #[serde(default = "default_page_image_density")]
    pub page_image_density: f32,
    #[serde(default = "default_segment_bbox_offset")]
    pub segment_bbox_offset: f32,
    #[serde(default = "default_page_limit")]
    pub page_limit: i32,
}

fn default_version() -> String {
    "1.0.3".to_string()
}

fn default_queue_preprocess() -> String {
    "preprocess".to_string()
}

fn default_queue_fast() -> String {
    "fast".to_string()
}

fn default_queue_high_quality() -> String {
    "high-quality".to_string()
}

fn default_queue_postprocess() -> String {
    "postprocess".to_string()
}

fn default_queue_ocr() -> String {
    "ocr".to_string()
}

fn default_queue_structured_extract() -> String {
    "structured-extract".to_string()
}

fn default_pdla_url() -> String {
    "http://localhost:8002".to_string()
}

fn default_pdla_fast_url() -> String {
    "http://localhost:8002".to_string()
}

fn default_general_ocr_url() -> Option<String> {
    Some("http://localhost:8003".to_string())
}

fn default_table_ocr_url() -> Option<String> {
    Some("http://localhost:8004".to_string())
}

fn default_server_url() -> String {
    "http://localhost:8000".to_string()
}

fn default_s3_bucket() -> String {
    "chunkr".to_string()
}

fn default_batch_size() -> i32 {
    300
}

fn default_ocr_concurrency() -> usize {
    24
}

fn default_ocr_confidence_threshold() -> f32 {
    0.85
}

fn default_pdf_density() -> f32 {
    72.0
}

fn default_page_image_density() -> f32 {
    150.0
}

fn default_segment_bbox_offset() -> f32 {
    1.0
}

fn default_page_limit() -> i32 {
    500
}

fn default_task_expiration() -> Option<Duration> {
    None
}

mod duration_seconds {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match duration {
            Some(d) => serializer.serialize_u64(d.as_secs()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<String> = Option::deserialize(deserializer)?;
        match value {
            Some(s) if !s.is_empty() => s
                .parse::<u64>()
                .map(|secs| Some(Duration::from_secs(secs)))
                .map_err(serde::de::Error::custom),
            _ => Ok(None),
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("EXTRACTION")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
