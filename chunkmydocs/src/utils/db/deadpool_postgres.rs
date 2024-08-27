use config::{ Config as ConfigTrait, ConfigError };
pub use deadpool_postgres::{ Client, Pool };
use deadpool_postgres::Runtime;
use dotenvy::dotenv_override;
use serde::Deserialize;
pub use tokio_postgres::Error;
use openssl::ssl::{ SslConnector, SslMethod, SslVerifyMode };
use postgres_openssl::MakeTlsConnector;

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
    // Create an SSL connector
    let mut builder = SslConnector::builder(SslMethod::tls()).unwrap();
    builder.set_verify(SslVerifyMode::NONE);
    let connector = MakeTlsConnector::new(builder.build());

    // Use the SSL connector when creating the pool
    cfg.pg.create_pool(Some(Runtime::Tokio1), connector).unwrap()
}
