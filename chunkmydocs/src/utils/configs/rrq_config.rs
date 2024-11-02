use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_url")]
    pub url: String,
    #[serde(default = "default_api_key")]
    pub api_key: String,
}

fn default_url() -> String {
    "http://localhost:8005".to_string()
}

fn default_api_key() -> String {
    "1234567890".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("RRQ").separator("__"))
            .build()?
            .try_deserialize()
    }
}
