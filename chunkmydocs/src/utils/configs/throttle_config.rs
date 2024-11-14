use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_general_ocr_rate_limit")]
    pub general_ocr_rate_limit: f32,
    #[serde(default = "default_llm_ocr_rate_limit")]
    pub llm_ocr_rate_limit: f32,
}

fn default_general_ocr_rate_limit() -> f32 {
    6.0
}

fn default_llm_ocr_rate_limit() -> f32 {
    200.0
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
