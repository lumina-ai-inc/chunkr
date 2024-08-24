use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv;
use reqwest::{multipart, Client};
use serde::Deserialize;
use serde_json::json;
use std::error::Error;
use std::io::copy;
use std::sync::Once;
use tempfile::NamedTempFile;

static INIT: Once = Once::new();

#[derive(Debug, Deserialize)]
struct Config {
    url: String,
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        INIT.call_once(|| {
            dotenv().ok();
        });

        ConfigTrait::builder()
            .add_source(
                config::Environment::default()
                    .prefix("STORAGE")
                    .separator("__"),
            )
            .build()?
            .try_deserialize()
    }
}

pub async fn download_to_tempfile(
    location: &str,
) -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let client = Client::new();
    let download_url = format!("{}/download", config.url);
    let response = client
        .post(&download_url)
        .json(&serde_json::json!({
            "location": location,
            "expires_in": null
        }))
        .send()
        .await?
        .error_for_status()?;

    let unsigned_url = response.text().await?;

    let mut temp_file = NamedTempFile::new()?;
    let content = client.get(&unsigned_url).send().await?.bytes().await?;
    copy(&mut content.as_ref(), &mut temp_file)?;

    Ok(temp_file)
}

pub async fn upload_to_s3(
    s3_path: &str,
    file_name: &str,
    buffer: Vec<u8>,
    expiration: Option<&str>,
) -> Result<bool, Box<dyn Error>> {
    let config = Config::from_env()?;
    let client = Client::new();

    let upload_url = format!("{}/upload", config.url);
    println!("Uploading to S3: {:?}", upload_url);
    let metadata = json!({
        "location": s3_path,
        "expiration": expiration
    });
    println!("Uploading to S3: {:?}", metadata);
    let form = multipart::Form::new()
        .part(
            "metadata",
            multipart::Part::text(serde_json::to_string(&metadata)?)
                .mime_str("application/json")?,
        )
        .part(
            "file",
            multipart::Part::bytes(buffer).file_name(file_name.to_string()),
        );

    let response = client
        .post(&upload_url)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    println!("Upload response status: {:?}", response.status());
    Ok(response.status().is_success())
}
