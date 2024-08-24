use config::{Config as ConfigTrait, ConfigError};
pub use deadpool_redis::{
    redis::{cmd, Pipeline, RedisError, RedisResult},
    Connection, Pool,
};
use deadpool_redis::{Config as RedisConfig, Runtime};
use dotenvy::dotenv;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Config {
    redis: RedisConfig,
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::env;
//     use uuid::Uuid;
//     use deadpool_redis::redis::cmd;

//     #[tokio::test]
//     async fn test_redis_pool() {
//         dotenv().ok();
//         env::var("REDIS__URL").expect("REDIS__URL must be set in .env file for tests");

//         let pool = create_pool();
//         let mut conn = pool.get().await.expect("Failed to get connection from pool");

//         let test_key = format!("test_key_{}", Uuid::new_v4());
//         let test_value = "42";

//         cmd("SET")
//             .arg(&[&test_key, test_value])
//             .query_async::<_, ()>(&mut conn).await
//             .expect("Failed to set value");

//         let value: String = cmd("GET")
//             .arg(&[&test_key])
//             .query_async(&mut conn).await
//             .expect("Failed to get value");

//         assert_eq!(value, test_value);

//         // Clean up: remove the test key
//         cmd("DEL")
//             .arg(&[&test_key])
//             .query_async::<_, ()>(&mut conn).await
//             .expect("Failed to delete test key");
//     }
// }
