use config::{Config as ConfigTrait, ConfigError};
use deadpool_postgres::Runtime;
pub use deadpool_postgres::{Client, Pool, PoolConfig};
use dotenvy::dotenv_override;
use openssl::ssl::{SslConnector, SslMethod, SslVerifyMode};
use postgres_openssl::MakeTlsConnector;
use serde::Deserialize;
use std::time::Duration;
pub use tokio_postgres::Error;

#[derive(Debug, Deserialize, Default)]
pub struct PoolSettings {
    pub timeout_wait_secs: Option<u64>,
    pub timeout_create_secs: Option<u64>,
    pub timeout_recycle_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub pg: deadpool_postgres::Config,
    #[serde(default)]
    pub pool: PoolSettings,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();
        ConfigTrait::builder()
            .add_source(config::Environment::default().separator("__"))
            .build()?
            .try_deserialize()
    }
}

pub fn create_pool() -> Pool {
    let mut cfg = Config::from_env().unwrap();
    cfg.pg.connect_timeout = Some(Duration::from_secs(10));

    // Set pool configuration to prevent hanging
    if cfg.pg.pool.is_none() {
        cfg.pg.pool = Some(PoolConfig::default());
    }
    if let Some(ref mut pool_config) = cfg.pg.pool {
        if let Some(timeout_wait_secs) = cfg.pool.timeout_wait_secs {
            pool_config.timeouts.wait = Some(Duration::from_secs(timeout_wait_secs));
        }
        if let Some(timeout_create_secs) = cfg.pool.timeout_create_secs {
            pool_config.timeouts.create = Some(Duration::from_secs(timeout_create_secs));
        }
        if let Some(timeout_recycle_secs) = cfg.pool.timeout_recycle_secs {
            pool_config.timeouts.recycle = Some(Duration::from_secs(timeout_recycle_secs));
        }
    }

    let mut builder = SslConnector::builder(SslMethod::tls()).unwrap();
    builder.set_verify(SslVerifyMode::NONE);
    let connector = MakeTlsConnector::new(builder.build());

    cfg.pg
        .create_pool(Some(Runtime::Tokio1), connector)
        .unwrap()
}
