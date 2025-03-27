use crate::configs::postgres_config::{create_pool, Pool};
use crate::configs::redis_config::{create_pool as create_redis_pool, Pool as RedisPool};
use crate::configs::s3_config::{create_client, create_external_client};
use crate::utils::rate_limit::init_throttle;
use aws_sdk_s3::Client as S3Client;
use futures::FutureExt;
use once_cell::sync::OnceCell;
use reqwest::Client as ReqwestClient;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::new();
static S3_CLIENT: OnceCell<S3Client> = OnceCell::new();
static S3_EXTERNAL_CLIENT: OnceCell<S3Client> = OnceCell::new();
static PG_POOL: OnceCell<Pool> = OnceCell::new();
static REDIS_POOL: OnceCell<RedisPool> = OnceCell::new();

pub async fn initialize() {
    REQWEST_CLIENT.get_or_init(ReqwestClient::new);
    S3_CLIENT.get_or_init(|| {
        async { create_client().await.unwrap() }
            .now_or_never()
            .unwrap()
    });
    S3_EXTERNAL_CLIENT.get_or_init(|| {
        async { create_external_client().await.unwrap() }
            .now_or_never()
            .unwrap()
    });
    PG_POOL.get_or_init(|| async { create_pool() }.now_or_never().unwrap());
    init_throttle();
    REDIS_POOL.get_or_init(create_redis_pool);
    println!("Initialized clients and rate limiters");
}

pub fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT.get().unwrap()
}

pub fn get_s3_client() -> &'static S3Client {
    S3_CLIENT.get().unwrap()
}

pub fn get_external_s3_client() -> &'static S3Client {
    S3_EXTERNAL_CLIENT.get().unwrap()
}

pub async fn get_pg_client() -> Result<deadpool_postgres::Client, deadpool_postgres::PoolError> {
    PG_POOL.get().unwrap().get().await
}

pub fn get_redis_pool() -> &'static RedisPool {
    REDIS_POOL.get().unwrap()
}
