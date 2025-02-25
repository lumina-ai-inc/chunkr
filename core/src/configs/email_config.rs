use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_server_url")]
    pub server_url: String,
}

fn default_server_url() -> String {
    "http://localhost:8000".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("EMAIL")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
