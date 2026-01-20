use aws_credential_types::Credentials;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::{Client, Config as S3Config};
use config::{Config as ConfigTrait, ConfigError};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    access_key: String,
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    pub presigned_url_endpoint: Option<String>,
    region: String,
    secret_key: String,
}

pub struct ExternalS3Client(pub Client);

fn default_endpoint() -> String {
    "https://s3.amazonaws.com".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        // #region agent log H5: Track config loading flow
        use std::fs::OpenOptions;use std::io::Write;let log_path="/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log";let mut f=OpenOptions::new().create(true).append(true).open(log_path).ok();if let Some(ref mut file)=f{let endpoint_before=std::env::var("AWS__ENDPOINT").unwrap_or_else(|_| "NOT_SET".to_string());let _=writeln!(file,r#"{{"sessionId":"debug-session","runId":"config-load","hypothesisId":"H5","location":"s3_config.rs:28","message":"Config::from_env called","data":{{"AWS__ENDPOINT_from_env":"{}","is_err":{}}},"timestamp":{}}}"#,endpoint_before,std::env::var("AWS__ENDPOINT").is_err(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());}
        // #endregion
        
        // For local development: Environment variables should take precedence over .env files
        // Only load .env if environment variables are NOT already set
        let should_load_dotenv = std::env::var("AWS__ENDPOINT").is_err();
        
        // #region agent log H6: Track dotenv decision
        if let Some(ref mut file)=f{let _=writeln!(file,r#"{{"sessionId":"debug-session","runId":"config-load","hypothesisId":"H6","location":"s3_config.rs:38","message":"Dotenv decision","data":{{"should_load_dotenv":{},"current_working_dir":"{}"}},"timestamp":{}}}"#,should_load_dotenv,std::env::current_dir().unwrap_or_default().display(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());}
        // #endregion
        
        if should_load_dotenv {
            dotenvy::dotenv().ok();
        }
        
        // #region agent log H7: After dotenv, before config builder
        if let Some(ref mut file)=f{let endpoint_after=std::env::var("AWS__ENDPOINT").unwrap_or_else(|_| "NOT_SET".to_string());let _=writeln!(file,r#"{{"sessionId":"debug-session","runId":"config-load","hypothesisId":"H7","location":"s3_config.rs:48","message":"After dotenv logic","data":{{"AWS__ENDPOINT_from_env":"{}","dotenv_loaded":{}}},"timestamp":{}}}"#,endpoint_after,should_load_dotenv,std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());}
        // #endregion
        
        let config_result = ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("AWS").separator("__"))
            .build()?
            .try_deserialize::<Self>()?;
        
        // #region agent log H8: Final config value
        if let Some(ref mut file)=f{let _=writeln!(file,r#"{{"sessionId":"debug-session","runId":"config-load","hypothesisId":"H8","location":"s3_config.rs:58","message":"Config deserialized","data":{{"config_endpoint":"{}","config_region":"{}"}},"timestamp":{}}}"#,config_result.endpoint,config_result.region,std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());}
        // #endregion
        
        Ok(config_result)
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
