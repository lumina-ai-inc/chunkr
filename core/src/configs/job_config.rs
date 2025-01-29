use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default = "default_task_timeout")]
    pub task_timeout: u32,
    pub expiration_time: Option<i32>,
    #[serde(default = "default_job_interval")]
    pub job_interval: u64,
}

fn default_job_interval() -> u64 {
    600
}

fn default_task_timeout() -> u32 {
    1800
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();
        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("JOB").separator("__"))
            .build()?
            .try_deserialize()
    }
}
