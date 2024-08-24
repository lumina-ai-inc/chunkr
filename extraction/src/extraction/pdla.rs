use super::extraction_config::Config;
use super::pdf::split_pdf;
use crate::models::extraction::extraction::ModelInternal;
use reqwest::{multipart, Client as ReqwestClient};
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use tempdir::TempDir;
use tempfile::NamedTempFile;
use tokio::sync::OnceCell;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT
        .get_or_init(|| async { ReqwestClient::new() })
        .await
}

async fn call_pdla_api(
    url: &str,
    file_path: &Path,
    fast: bool,
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
        .part("file", part)
        .text("fast", fast.to_string());

    let response = client
        .post(url)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;
    Ok(response.text().await?)
}

async fn handle_fast_requests(file_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let url = config.pdla_fast_url;
    call_pdla_api(&url, file_path, true).await
}

async fn handle_high_quality_requests(
    file_path: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let url = config.pdla_url;
    call_pdla_api(&url, file_path, false).await
}

async fn process_file(
    file_path: &Path,
    batch_size: Option<i32>,
    model: ModelInternal,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut temp_files: Vec<PathBuf> = vec![];
    let temp_dir = TempDir::new("split_pdf")?;
    if let Some(batch_size) = batch_size {
        temp_files = split_pdf(file_path, batch_size as usize, temp_dir.path())?;
    } else {
        temp_files.push(file_path.to_path_buf());
    }
    let mut combined_output = Vec::new();
    let mut page_offset = 0;

    for temp_file in &temp_files {
        let json_output = if model == ModelInternal::PdlaFast {
            handle_fast_requests(&temp_file).await?
        } else if model == ModelInternal::Pdla {
            handle_high_quality_requests(&temp_file).await?
        } else {
            return Err(format!("Invalid model: {}", model).into());
        };

        let mut batch_output: Vec<Value> = serde_json::from_str(&json_output)?;
        for item in &mut batch_output {
            if let Some(page_number) = item.get_mut("page_number") {
                *page_number = serde_json::json!(page_number.as_i64().unwrap() + page_offset);
            }
        }
        combined_output.extend(batch_output);
        page_offset += batch_size.unwrap_or(1) as i64;
    }

    Ok(serde_json::to_string(&combined_output)?)
}

pub async fn pdla_extraction(
    file_path: &Path,
    model: ModelInternal,
    batch_size: Option<i32>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let json_output = process_file(file_path, batch_size, model).await?;

    let mut output_temp_file = NamedTempFile::new()?;
    output_temp_file.write_all(json_output.as_bytes())?;

    Ok(output_temp_file.into_temp_path().keep()?.to_path_buf())
}
