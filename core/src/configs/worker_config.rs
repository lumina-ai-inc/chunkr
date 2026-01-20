use config::{Config as ConfigTrait, ConfigError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum FileUrlFormat {
    Base64,
    Url,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_file_url_format")]
    pub file_url_format: FileUrlFormat,
    #[serde(default = "default_general_ocr_url")]
    pub general_ocr_url: Option<String>,
    #[serde(default = "default_high_res_scaling_factor")]
    pub high_res_scaling_factor: f32,
    #[serde(default = "default_ocr_confidence_threshold")]
    pub ocr_confidence_threshold: f32,
    #[serde(default = "default_page_limit")]
    pub page_limit: i32,
    #[serde(default = "default_queue_task")]
    pub queue_task: String,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_s3_bucket")]
    pub s3_bucket: String,
    #[serde(default = "default_segmentation_padding")]
    pub segmentation_padding: f32,
    #[serde(default = "default_segmentation_url")]
    pub segmentation_url: String,
    #[serde(default = "default_server_url")]
    pub server_url: String,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_file_url_format() -> FileUrlFormat {
    FileUrlFormat::Base64
}

fn default_general_ocr_url() -> Option<String> {
    Some("http://localhost:8002".to_string())
}

fn default_high_res_scaling_factor() -> f32 {
    2.0
}

fn default_ocr_confidence_threshold() -> f32 {
    0.85
}

fn default_page_limit() -> i32 {
    10000
}

fn default_queue_task() -> String {
    "task".to_string()
}

fn default_max_retries() -> u32 {
    3
}

fn default_s3_bucket() -> String {
    "chunkr".to_string()
}

fn default_segmentation_padding() -> f32 {
    1.0
}

fn default_segmentation_url() -> String {
    "http://localhost:8001".to_string()
}

fn default_server_url() -> String {
    "http://localhost:8000".to_string()
}

fn default_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        // #region agent log H11: WorkerConfig loading
        use std::fs::OpenOptions;use std::io::Write;let log_path="/Users/harishmaiya/Documents/GitHub/data-extract/.cursor/debug.log";let mut f=OpenOptions::new().create(true).append(true).open(log_path).ok();if let Some(ref mut file)=f{let _=writeln!(file,r#"{{"sessionId":"debug-session","runId":"worker-config","hypothesisId":"H11","location":"worker_config.rs:88","message":"WorkerConfig::from_env called","data":{{"AWS__ENDPOINT_before":"{}","will_call_dotenv":{}}},"timestamp":{}}}"#,std::env::var("AWS__ENDPOINT").unwrap_or_else(|_| "NOT_SET".to_string()),std::env::var("AWS__ENDPOINT").is_err(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());}
        // #endregion
        
        // For local development: Environment variables should take precedence over .env files
        // Only load .env if critical environment variables are NOT already set
        if std::env::var("AWS__ENDPOINT").is_err() {
            dotenvy::dotenv().ok();
        }
        
        // #region agent log H12: WorkerConfig after dotenv
        if let Some(ref mut file)=f{let _=writeln!(file,r#"{{"sessionId":"debug-session","runId":"worker-config","hypothesisId":"H12","location":"worker_config.rs:99","message":"WorkerConfig after dotenv check","data":{{"AWS__ENDPOINT_after":"{}"}},"timestamp":{}}}"#,std::env::var("AWS__ENDPOINT").unwrap_or_else(|_| "NOT_SET".to_string()),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());}
        // #endregion

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("WORKER")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}
