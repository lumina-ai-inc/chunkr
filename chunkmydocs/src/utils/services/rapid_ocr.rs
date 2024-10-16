use crate::utils::configs::extraction_config::Config;
use reqwest::{ multipart, Client as ReqwestClient };
use serde_json::Value;
use std::{ fs, path::Path };
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT.get_or_init(|| async { ReqwestClient::new() }).await
}

// TODO: Add rapid ocr payload model
pub async fn call_rapid_ocr_api(file_path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let client = get_reqwest_client().await;
    let url = format!("{}/{}", config.rapid_ocr_url.unwrap(), "ocr");

    let file_name = file_path
        .file_name()
        .ok_or_else(|| format!("Invalid file name: {:?}", file_path))?
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 file name: {:?}", file_path))?
        .to_string();

    let file_fs = fs::read(file_path).expect("Failed to read file");
    let part = multipart::Part::bytes(file_fs).file_name(file_name);

    let form = multipart::Form::new().part("file", part);

    let response = client
        .post(url)
        .multipart(form)
        .timeout(std::time::Duration::from_secs(30))
        .send().await?
        .error_for_status()?;

    Ok(response.json().await?)
}
