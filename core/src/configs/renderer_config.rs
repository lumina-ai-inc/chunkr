use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_headless")]
    pub headless: bool,
    #[serde(default = "default_sandbox")]
    pub sandbox: bool,
}

fn default_headless() -> bool {
    true
}

fn default_sandbox() -> bool {
    true
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("RENDERER")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
