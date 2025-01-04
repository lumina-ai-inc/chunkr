use crate::configs::postgres_config::{create_pool, Pool};
use crate::configs::s3_config::{create_client, create_external_client};
use aws_sdk_s3::Client as S3Client;
use futures::FutureExt;
use once_cell::sync::OnceCell;
use reqwest::Client as ReqwestClient;
static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::new();
static S3_CLIENT: OnceCell<S3Client> = OnceCell::new();
static S3_EXTERNAL_CLIENT: OnceCell<S3Client> = OnceCell::new();
static PG_POOL: OnceCell<Pool> = OnceCell::new();

pub async fn initialize_clients() {
    REQWEST_CLIENT.get_or_init(|| ReqwestClient::new());
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
    println!("Clients initialized");
}

pub fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT
        .get()
        .expect("Reqwest client not initialized")
}

pub fn get_s3_client() -> &'static S3Client {
    S3_CLIENT.get().expect("S3 client not initialized")
}

pub fn get_external_s3_client() -> &'static S3Client {
    S3_EXTERNAL_CLIENT
        .get()
        .expect("External S3 client not initialized")
}

pub async fn get_pg_client() -> Result<deadpool_postgres::Client, deadpool_postgres::PoolError> {
    Ok(PG_POOL
        .get()
        .expect("Postgres pool not initialized")
        .get()
        .await?)
}
