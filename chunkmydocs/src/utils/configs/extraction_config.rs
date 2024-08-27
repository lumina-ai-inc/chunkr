use config::{ Config as ConfigTrait, ConfigError };
use dotenvy::dotenv_override;
use serde::{ Deserialize, Serialize };
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub version: String,
    pub extraction_queue: String,
    pub grobid_url: String,
    pub pdla_url: String,
    pub pdla_fast_url: String,
    pub table_ocr_url: String,
    #[serde(with = "duration_seconds")]
    pub task_expiration: Option<Duration>,
    pub s3_bucket: String,
    pub batch_size: i32,
    pub base_url: String,
}

mod duration_seconds {
    use serde::{ Deserialize, Deserializer, Serializer };
    use std::time::Duration;

    pub fn serialize<S>(duration: &Option<Duration>, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        match duration {
            Some(d) => serializer.serialize_u64(d.as_secs()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Duration>, D::Error>
        where D: Deserializer<'de>
    {
        let seconds: Option<u64> = Option::deserialize(deserializer)?;
        Ok(seconds.map(Duration::from_secs))
    }
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("EXTRACTION").separator("__"))
            .build()?
            .try_deserialize()
    }
}
