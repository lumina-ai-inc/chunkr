use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub time: Option<i32>,
    #[serde(default = "default_job_interval")]
    pub job_interval: u64,
}

fn default_job_interval() -> u64 {
    600
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
