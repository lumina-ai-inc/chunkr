use config::{Config as ConfigTrait, ConfigError};
pub use deadpool_postgres::{Client, Pool};
use deadpool_postgres::{Config as PgConfig, Runtime};
use dotenvy::dotenv;
use serde::Deserialize;
pub use tokio_postgres::Error;

// Add these new imports
use openssl::ssl::{SslConnector, SslMethod, SslVerifyMode};
use postgres_openssl::MakeTlsConnector;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub pg: PgConfig,
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
    // Create an SSL connector
    let mut builder = SslConnector::builder(SslMethod::tls()).unwrap();
    builder.set_verify(SslVerifyMode::NONE);
    let connector = MakeTlsConnector::new(builder.build());

    // Use the SSL connector when creating the pool
    cfg.pg.create_pool(Some(Runtime::Tokio1), connector).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[tokio::test]
    async fn test_pg_pool() {
        // Load .env file
        dotenv().ok();

        // Ensure PG__URL is set in the .env file
        env::var("PG__URL").expect("PG__URL must be set in .env file for tests");

        // Create the pool
        let pool = create_pool();

        // Try to get a connection from the pool
        let client = pool.get().await.expect("Failed to get client from pool");
        let result = client
            .query_one("SELECT 1", &[])
            .await
            .expect("Query failed");
        assert_eq!(result.get::<_, i32>(0), 1);
    }
}
