use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub version: String,
    pub queue_preprocess: String,
    pub queue_fast: String,
    pub queue_high_quality: String,
    pub queue_postprocess: String,
    pub queue_ocr: String,
    pub queue_structured_extract: String,
    pub pdla_url: String,
    pub pdla_fast_url: String,
    pub rapid_ocr_url: String,
    pub table_structure_url: String,
    #[serde(with = "duration_seconds")]
    pub task_expiration: Option<Duration>,
    pub s3_bucket: String,
    pub batch_size: i32,
    pub base_url: String,
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

fn default_ocr_concurrency() -> usize {
    10
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
    5.0
}

fn default_page_limit() -> i32 {
    500
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
