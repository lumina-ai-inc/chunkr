use config::{Config as ConfigTrait, ConfigError};
use deadpool_postgres::Runtime;
pub use deadpool_postgres::{Client, Pool};
use dotenvy::dotenv_override;
use openssl::ssl::{SslConnector, SslMethod, SslVerifyMode};
use postgres_openssl::MakeTlsConnector;
use serde::Deserialize;
pub use tokio_postgres::Error;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub pg: deadpool_postgres::Config,
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
    let cfg = Config::from_env().unwrap();
    let mut builder = SslConnector::builder(SslMethod::tls()).unwrap();
    builder.set_verify(SslVerifyMode::NONE);
    let connector = MakeTlsConnector::new(builder.build());

    cfg.pg
        .create_pool(Some(Runtime::Tokio1), connector)
        .unwrap()
}
