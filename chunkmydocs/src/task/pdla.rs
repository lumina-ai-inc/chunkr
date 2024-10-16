use crate::models::server::extract::SegmentationModel;
use crate::utils::configs::extraction_config::Config;
use reqwest::{multipart, Client as ReqwestClient};
use std::{fs, path::Path};
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT
        .get_or_init(|| async { ReqwestClient::new() })
        .await
}

async fn call_pdla_api(
    url: &str,
    file_path: &Path
) -> Result<String, Box<dyn std::error::Error>> {
    let client = get_reqwest_client().await;

    let file_name = file_path
        .file_name()
        .ok_or_else(|| format!("Invalid file name: {:?}", file_path))?
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 file name: {:?}", file_path))?
        .to_string();

    let file_fs = fs::read(file_path).expect("Failed to read file");
    let part = multipart::Part::bytes(file_fs).file_name(file_name);

    let form = multipart::Form::new()
        .part("file", part);

    let response = client
        .post(url)
        .multipart(form)
        .timeout(std::time::Duration::from_secs(10000)) 
        .send()
        .await?
        .error_for_status()?;
    Ok(response.text().await?)
}

async fn handle_fast_requests(file_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let url = format!("{}/analyze/fast", config.pdla_fast_url);
    call_pdla_api(&url, file_path).await
}

async fn handle_high_quality_requests(
    file_path: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let url = format!("{}/analyze/high-quality", config.pdla_url);
    call_pdla_api(&url, file_path).await
}

async fn process_file(
    file_path: &Path,
    model: SegmentationModel,
) -> Result<String, Box<dyn std::error::Error>> {
    let json_output = if model == SegmentationModel::PdlaFast {
        handle_fast_requests(file_path).await?
    } else if model == SegmentationModel::Pdla {
        handle_high_quality_requests(file_path).await?
    } else {
        return Err(format!("Invalid model: {}", model).into());
    };

    Ok(json_output)
}

pub async fn pdla_extraction(
    file_path: &Path,
    model: SegmentationModel,
) -> Result<String, Box<dyn std::error::Error>> {
    let json_output = process_file(file_path, model).await?;
    Ok(json_output)
}
