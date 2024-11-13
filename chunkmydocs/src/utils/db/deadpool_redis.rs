use config::{Config as ConfigTrait, ConfigError};
pub use deadpool_redis::{
    redis::{cmd, Pipeline, RedisError, RedisResult},
    Connection, Pool,
};
use deadpool_redis::{Config as RedisConfig, Runtime};
use dotenvy::dotenv;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub redis: RedisConfig,
    #[serde(default = "default_api_key")]
    pub api_key: String,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_api_key() -> String {
    "1234567890".to_string()
}

fn default_version() -> String {
    "1.0.0".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        ConfigTrait::builder()
            .add_source(config::Environment::default().separator("__"))
            .build()?
            .try_deserialize()
    }
}

pub fn create_pool() -> Pool {
    dotenv().ok();
    let cfg = Config::from_env().unwrap();
    cfg.redis.create_pool(Some(Runtime::Tokio1)).unwrap()
}
