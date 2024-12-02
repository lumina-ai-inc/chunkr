use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub expiration_time: Option<u64>,
    #[serde(default = "default_job_interval")]
    pub job_interval: u64,
}

fn default_job_interval() -> u64 {
    1
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();
        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("EXPIRATION")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
