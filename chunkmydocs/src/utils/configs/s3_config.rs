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
    pub presigned_url_endpoint: Option<String>,
    region: String,
}

pub struct ExternalS3Client(pub Client);

fn default_endpoint() -> String {
    "https://s3.amazonaws.com".to_string()
}

fn default_presigned_url_endpoint() -> Option<String> {
    None
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

fn get_aws_config(external: bool) -> Result<S3Config, ConfigError> {
    let config = Config::from_env()?;
    let creds = Credentials::from_keys(config.access_key, config.secret_key, None);
    let endpoint_url = if external {
        config.presigned_url_endpoint.unwrap_or(config.endpoint)
    } else {
        config.endpoint
    };
    let aws_config = S3Config::builder()
        .credentials_provider(creds)
        .region(Region::new(config.region))
        .force_path_style(true)
        .endpoint_url(endpoint_url)
        .build();

    Ok(aws_config)
}

pub async fn create_client() -> Result<Client, ConfigError> {
    let aws_config = get_aws_config(false)?;
    Ok(aws_sdk_s3::Client::from_conf(aws_config))
}

pub async fn create_external_client() -> Result<Client, ConfigError> {
    let aws_config = get_aws_config(true)?;
    Ok(aws_sdk_s3::Client::from_conf(aws_config))
}
