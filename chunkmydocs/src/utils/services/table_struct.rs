use crate::models::workers::table_ocr::{TableStructure, TableStructureResponse};
use crate::utils::configs::extraction_config::Config;
use reqwest::multipart;
use std::error::Error;
use std::{fs, path::Path};
pub async fn recognize_table(file_path: &Path) -> Result<Vec<TableStructure>, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let config = Config::from_env()?;
    let url = format!("{}/predict/table", &config.table_structure_url);

    let file_name = file_path
        .file_name()
        .ok_or_else(|| format!("Invalid file name: {:?}", file_path))?
        .to_str()
        .ok_or_else(|| format!("Non-UTF8 file name: {:?}", file_path))?
        .to_string();

    let file_fs = fs::read(file_path).expect("Failed to read file");
    let part = multipart::Part::bytes(file_fs).file_name(file_name);
    let form = multipart::Form::new().part("files", part);
    let response = client
        .post(&url)
        .multipart(form)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?
        .error_for_status()?;

    let table_struct_response: TableStructureResponse = response.json().await?;
    Ok(table_struct_response.result)
}
#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_recognize_table() {
        let image_path = std::path::Path::new(
            "/Users/ishaankapoor/Startup/chunk-my-docs/chunkmydocs/input/test.jpg",
        );

        let result = recognize_table(image_path).await;

        assert!(result.is_ok(), "recognize_table failed: {:?}", result.err());
    }
}
