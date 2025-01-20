use crate::configs::azure_config;
use crate::models::chunkr::azure::AzureAnalysisResponse;
use crate::models::chunkr::output::Chunk;
use crate::utils::clients;
use crate::utils::retry::retry_with_backoff;
use base64::{engine::general_purpose, Engine as _};
use serde_json;
use std::error::Error;
use std::fs;
use tempfile::NamedTempFile;

async fn azure_analysis(temp_file: &NamedTempFile) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let azure_config = azure_config::Config::from_env()?;
    let api_version = azure_config.api_version;
    let endpoint = azure_config.endpoint;
    let key = azure_config.key;
    let model_id = azure_config.model_id;
    let client = clients::get_reqwest_client();

    let url = format!(
        "{}/documentintelligence/documentModels/{}:analyze?_overload=analyzeDocument&api-version={}",
        endpoint.trim_end_matches('/'),
        model_id,
        api_version
    );

    let file_content = fs::read(temp_file.path())?;
    let base64_content = general_purpose::STANDARD.encode(&file_content);

    let request_body = serde_json::json!({
        "base64Source": base64_content
    });

    let response = client
        .post(&url)
        .header("Ocp-Apim-Subscription-Key", key.clone())
        .json(&request_body)
        .send()
        .await?;

    if response.status() == 202 {
        let operation_location = response
            .headers()
            .get("operation-location")
            .ok_or("No operation-location header found")?
            .to_str()?;

        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

            let status_response = client
                .get(operation_location)
                .header("Ocp-Apim-Subscription-Key", key.clone())
                .send()
                .await?
                .error_for_status()?;

            let azure_response: AzureAnalysisResponse = status_response.json().await?;

            match azure_response.status.as_str() {
                "succeeded" => {
                    let chunks = azure_response.to_chunks()?;
                    return Ok(chunks);
                }
                "failed" => return Err("Analysis failed".into()),
                "running" | "notStarted" => {
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    continue;
                }
                _ => return Err("Unknown status".into()),
            }
        }
    }

    Err("Unknown status".into())
}

pub async fn perform_azure_analysis(
    temp_file: &NamedTempFile,
) -> Result<Vec<Chunk>, Box<dyn Error>> {
    Ok(retry_with_backoff(|| async { azure_analysis(temp_file).await }).await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::clients::initialize;
    use std::path::Path;

    #[tokio::test]
    async fn test_azure_analysis() {
        initialize().await;
        let test_file_path = Path::new("./input/test.pdf");
        let temp_file = NamedTempFile::new().unwrap();
        fs::copy(test_file_path, temp_file.path()).unwrap();
        let result = azure_analysis(&temp_file)
            .await
            .expect("Azure analysis failed");
        let json = serde_json::to_string_pretty(&result).unwrap();
        fs::write("azure-analysis-response.json", json).unwrap();
    }
}
