use config::{Config as ConfigTrait, ConfigError};
use deadpool_redis::{Config as RedisConfig, Pool, Runtime};
use dotenvy::dotenv;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub redis: RedisConfig,
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
    // For local development: Environment variables should take precedence over .env files
    // Only load .env if environment variables are NOT already set
    if std::env::var("REDIS__URL").is_err() {
        dotenv().ok();
    }
    let cfg = Config::from_env().unwrap();
    cfg.redis.create_pool(Some(Runtime::Tokio1)).unwrap()
}
