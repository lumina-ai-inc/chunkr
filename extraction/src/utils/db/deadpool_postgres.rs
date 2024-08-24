use deadpool_postgres::{Runtime, Config as PgConfig};
use dotenvy::dotenv;
use serde::Deserialize;
use config::{Config as ConfigTrait, ConfigError};
pub use deadpool_postgres::{Pool, Client};
pub use tokio_postgres::{Error, NoTls};

#[derive(Debug, Deserialize)]
struct Config {
    pg: PgConfig,
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
    cfg.pg.create_pool(Some(Runtime::Tokio1), NoTls).unwrap()
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
        let result = client.query_one("SELECT 1", &[]).await.expect("Query failed");
        assert_eq!(result.get::<_, i32>(0), 1);
    }
}