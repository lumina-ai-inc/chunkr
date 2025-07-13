use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_azure_analysis_rate_limit")]
    pub azure_analysis_rate_limit: f32,
    #[serde(default = "default_azure_polling_rate_limit")]
    pub azure_polling_rate_limit: f32,
    #[serde(default = "default_azure_timeout")]
    pub azure_timeout: u64,
    #[serde(default = "default_general_ocr_batch_size")]
    pub general_ocr_batch_size: usize,
    #[serde(default = "default_general_ocr_rate_limit")]
    pub general_ocr_rate_limit: f32,
    #[serde(default = "default_general_ocr_timeout")]
    pub general_ocr_timeout: Option<u64>,
    #[serde(default = "default_llm_rate_limit")]
    pub llm_rate_limit: Option<f32>,
    #[serde(default = "default_llm_timeout")]
    pub llm_timeout: Option<u64>,
    #[serde(default = "default_segmentation_batch_size")]
    pub segmentation_batch_size: usize,
    #[serde(default = "default_segmentation_rate_limit")]
    pub segmentation_rate_limit: f32,
    #[serde(default = "default_segmentation_timeout")]
    pub segmentation_timeout: Option<u64>,
}

fn default_azure_analysis_rate_limit() -> f32 {
    15.0
}

fn default_azure_polling_rate_limit() -> f32 {
    45.0
}

fn default_azure_timeout() -> u64 {
    120
}

fn default_general_ocr_batch_size() -> usize {
    30
}

fn default_general_ocr_rate_limit() -> f32 {
    5.0
}

fn default_general_ocr_timeout() -> Option<u64> {
    None
}

fn default_llm_rate_limit() -> Option<f32> {
    None
}

fn default_llm_timeout() -> Option<u64> {
    Some(150)
}

fn default_segmentation_batch_size() -> usize {
    3
}

fn default_segmentation_rate_limit() -> f32 {
    5.0
}

fn default_segmentation_timeout() -> Option<u64> {
    None
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("THROTTLE")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
