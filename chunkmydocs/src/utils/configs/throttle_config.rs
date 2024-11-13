use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_general_ocr_concurrency")]
    pub general_ocr_concurrency: usize,
    #[serde(default = "default_vlm_ocr_concurrency")]
    pub vlm_ocr_concurrency: usize,
    #[serde(default = "default_vlm_ocr_rate_limit")]
    pub vlm_ocr_rate_limit: u32,
}

fn default_general_ocr_concurrency() -> usize {
    25
}

fn default_vlm_ocr_concurrency() -> usize {
    10000
}

fn default_vlm_ocr_rate_limit() -> u32 {
    200
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("RATE_LIMIT")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
