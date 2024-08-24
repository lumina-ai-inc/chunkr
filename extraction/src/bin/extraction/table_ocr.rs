use reqwest::{ multipart, Client as ReqwestClient };
use tokio::sync::OnceCell;
use models::api::extraction::ModelInternal;
use std::{ fs, io::Write, path::{ Path, PathBuf } };
use tempfile::NamedTempFile;
use serde_json::Value;
use tempdir::TempDir;
use crate::pdf::split_pdf;
use crate::extraction_config::Config;

static REQWEST_CLIENT: OnceCell<ReqwestClient> = OnceCell::const_new();

async fn get_reqwest_client() -> &'static ReqwestClient {
    REQWEST_CLIENT.get_or_init(|| async { ReqwestClient::new() }).await
}

async fn call_table_extraction_api(
    url: &str,
    file_path: &Path
) -> Result<String, Box<dyn std::error::Error>> {
    let client = get_reqwest_client().await;

    let file_name = file_path.file_name()
        .ok_or_else(|| format!("Invalid file name: {:?}", file_path))?
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 file name: {:?}", file_path))?
        .to_string();
    let file_fs = fs::read(file_path).expect("Failed to read file");
    let part = multipart::Part::bytes(file_fs).file_name(file_name);

    let response = client.post(url).multipart(form).send().await?.error_for_status()?;
    Ok(response.text().await?)
}

async fn handle_requests(
    file_path: &Path
) -> Result<String, Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let url = config.table_ocr_url;
    call_table_extraction_api(&url, file_path).await
}

pub async fn table_extraction_from_image(
    file_path: &Path,
    json_file_path: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let json_output = process_file(file_path, batch_size, model).await?;

    let mut output_temp_file = NamedTempFile::new()?;
    output_temp_file.write_all(json_output.as_bytes())?;

    Ok(output_temp_file.into_temp_path().keep()?.to_path_buf())
}