use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_general_ocr_url")]
    pub general_ocr_url: Option<String>,
    #[serde(default = "default_high_res_scaling_factor")]
    pub high_res_scaling_factor: f32,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_ocr_confidence_threshold")]
    pub ocr_confidence_threshold: f32,
    #[serde(default = "default_page_limit")]
    pub page_limit: i32,
    #[serde(default = "default_queue_task")]
    pub queue_task: String,
    #[serde(default = "default_rasterization_batch_size")]
    pub rasterization_batch_size: usize,
    #[serde(default = "default_s3_bucket")]
    pub s3_bucket: String,
    #[serde(default = "default_segmentation_padding")]
    pub segmentation_padding: f32,
    #[serde(default = "default_segmentation_url")]
    pub segmentation_url: String,
    #[serde(default = "default_server_url")]
    pub server_url: String,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_general_ocr_url() -> Option<String> {
    Some("http://localhost:8002".to_string())
}

fn default_high_res_scaling_factor() -> f32 {
    2.0
}

fn default_max_retries() -> u32 {
    3
}

fn default_ocr_confidence_threshold() -> f32 {
    0.85
}

fn default_page_limit() -> i32 {
    10000
}

fn default_rasterization_batch_size() -> usize {
    100
}

fn default_queue_task() -> String {
    "task".to_string()
}

fn default_s3_bucket() -> String {
    "chunkr".to_string()
}

fn default_segmentation_padding() -> f32 {
    1.0
}

fn default_segmentation_url() -> String {
    "http://localhost:8001".to_string()
}

fn default_server_url() -> String {
    "http://localhost:8000".to_string()
}

fn default_version() -> String {
    "1.1.0".to_string()
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
