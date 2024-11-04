use aws_credential_types::Credentials;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::{Client, Config as S3Config};
use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    access_key: String,
    secret_key: String,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_presigned_url_endpoint")]
    pub presigned_url_endpoint: String,
    region: String,
}

fn default_endpoint() -> String {
    "https://s3.amazonaws.com".to_string()
}

fn default_presigned_url_endpoint() -> String {
    "https://s3.amazonaws.com".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();
        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("AWS").separator("__"))
            .build()?
            .try_deserialize()
    }
}

pub async fn create_client() -> Result<Client, ConfigError> {
    let config = Config::from_env()?;
    let creds = Credentials::from_keys(config.access_key, config.secret_key, None);

    let aws_config = S3Config::builder()
        .credentials_provider(creds)
        .region(Region::new(config.region))
        .force_path_style(true)
        .endpoint_url(config.endpoint)
        .build();

    let client = aws_sdk_s3::Client::from_conf(aws_config);
    Ok(client)
}
