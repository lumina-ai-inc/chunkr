use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_api_version")]
    pub api_version: String,
    pub endpoint: String,
    pub key: String,
    #[serde(default = "default_model_id")]
    pub model_id: String,
}

fn default_api_version() -> String {
    "2024-11-30".to_string()
}

fn default_model_id() -> String {
    "prebuilt-layout".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("AZURE")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
