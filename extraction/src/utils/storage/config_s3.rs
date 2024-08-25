use aws_credential_types::Credentials;
use aws_sdk_s3::{ Config as S3Config, Client };
use aws_sdk_s3::config::Region;
use config::{ Config as ConfigTrait, ConfigError };
use dotenvy::dotenv;
use serde::Deserialize;
use std::sync::Once;

static INIT: Once = Once::new();

#[derive(Debug, Deserialize)]
pub struct Config {
    access_key: String,
    secret_key: String,
    region: Option<String>,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        INIT.call_once(|| {
            dotenv().ok();
        });

        ConfigTrait::builder()
<<<<<<< HEAD
            .add_source(config::Environment::default().prefix("AWS").separator("__"))
=======
            .add_source(config::Environment::default().separator("__"))
>>>>>>> 35f1168 (done with s3 config)
            .build()?
            .try_deserialize()
    }
}

// pub async fn create_client() -> Result<Client, ConfigError> {
//     let config = Config::from_env()?;
//     let creds = Credentials::from_keys(config.aws_access_key, config.aws_secret_key, None);
//     let region_provider = RegionProviderChain::first_try("us-west-1");
//     let config = S3Config::builder()
//         .credentials_provider(creds)
//         .region(region_provider.region().await.unwrap())
//         .build();
//     let client = aws_sdk_s3::Client::from_conf(config);
//     Ok(client)
// }

pub async fn create_client() -> Result<Client, ConfigError> {
    let config = Config::from_env()?;
<<<<<<< HEAD
    let creds = Credentials::from_keys(config.access_key, config.secret_key, None);
    let region = config.region.unwrap_or_else(|| "us-west-1".to_string());
=======
    let creds = Credentials::from_keys(config.aws_access_key, config.aws_secret_key, None);
    let region = config.aws_region.unwrap_or_else(|| "us-west-1".to_string());
>>>>>>> 35f1168 (done with s3 config)
    let config = S3Config::builder()
        .credentials_provider(creds)
        .region(Region::new(region))
        .build();
    let client = aws_sdk_s3::Client::from_conf(config);
    Ok(client)
}