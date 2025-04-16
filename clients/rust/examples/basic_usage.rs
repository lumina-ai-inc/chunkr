use std::time::Duration;
use dotenvy::dotenv;
use config::{Config, ConfigError, Environment, File};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use chunkr_ai::Client;
use serde::Deserialize;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_url")]
    pub url: String,
    pub api_key: String,
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout: u64,
    #[serde(default = "default_request_timeout")]
    pub request_timeout: u64,
}

fn default_url() -> String {
    "https://api.chunkr.ai".to_string()
}

fn default_connect_timeout() -> u64 {
    15
}

fn default_request_timeout() -> u64 {
    30
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("CHUNKR")
                    .separator("_"),
            )
            .build()?
            .try_deserialize()
    }
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = Config::from_env()?;
    
    // Create custom headers
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&config.api_key).unwrap(),
    );
    
    // Build a custom reqwest client with our headers and timeouts
    let client_with_custom_defaults = reqwest::ClientBuilder::new()
        .connect_timeout(Duration::from_secs(config.connect_timeout))
        .timeout(Duration::from_secs(config.request_timeout))
        .default_headers(headers)
        .build()
        .unwrap();
    
    // Initialize the Chunkr client with our custom reqwest client
    let client = Client::new_with_client(config.url, client_with_custom_defaults);
    
    // Make a request - for example, a health check
    let health_response = client.health_check().await?;
    println!("Health check response: {:?}", health_response);
    
    Ok(())
} 