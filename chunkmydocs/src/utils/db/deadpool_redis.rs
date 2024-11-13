use config::{Config as ConfigTrait, ConfigError};
pub use deadpool_redis::{
    redis::{Pipeline, RedisError, RedisResult},
    Connection, Pool,
};
use deadpool_redis::{Config as RedisConfig, Runtime};
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
    dotenv().ok();
    let cfg = Config::from_env().unwrap();
    cfg.redis.create_pool(Some(Runtime::Tokio1)).unwrap()
}
