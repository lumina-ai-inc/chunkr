use crate::configs::azure_config;
use crate::models::chunkr::azure::{AzureAnalysisResponse, DocumentAnalysisFeature};
use crate::models::chunkr::output::Chunk;
use crate::models::chunkr::upload::SegmentationStrategy;
use crate::utils::clients;
use crate::utils::retry::retry_with_backoff;
use base64::{engine::general_purpose, Engine as _};
use serde_json;
use std::error::Error;
use std::fs;
use tempfile::NamedTempFile;

async fn azure_analysis(
    temp_file: &NamedTempFile,
    features: Option<Vec<String>>,
    segmentation_strategy: SegmentationStrategy,
) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let azure_config = azure_config::Config::from_env()?;
    let api_version = azure_config.api_version;
    let endpoint = azure_config.endpoint;
    let key = azure_config.key;
    let model_id = azure_config.model_id;
    let client = clients::get_reqwest_client();

    let mut url = format!(
        "{}/documentintelligence/documentModels/{}:analyze?_overload=analyzeDocument&api-version={}",
        endpoint.trim_end_matches('/'),
        model_id,
        api_version
    );

    if let Some(features) = features {
        url.push_str("&features=");
        url.push_str(&features.join(","));
    }

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
                    let chunks = azure_response.to_chunks(segmentation_strategy)?;
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
    features: Option<Vec<DocumentAnalysisFeature>>,
    segmentation_strategy: SegmentationStrategy,
) -> Result<Vec<Chunk>, Box<dyn Error>> {
    let features_str = features.map(|f| {
        f.into_iter()
            .map(|feature| feature.as_str().to_string())
            .collect()
    });

    retry_with_backoff(|| async {
        azure_analysis(
            temp_file,
            features_str.clone(),
            segmentation_strategy.clone(),
        )
        .await
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::clients::initialize;
    use std::path::Path;

    #[tokio::test]
    async fn test_azure_analysis() {
        initialize().await;
        let test_file_path = Path::new("./input/test.png");
        let output_dir = Path::new("./output/azure");
        fs::create_dir_all(output_dir).unwrap();
        let temp_file = NamedTempFile::new().unwrap();
        fs::copy(test_file_path, temp_file.path()).unwrap();
        let result = azure_analysis(&temp_file, None, SegmentationStrategy::LayoutAnalysis)
            .await
            .expect("Azure analysis failed");
        let json = serde_json::to_string_pretty(&result).unwrap();
        fs::write(output_dir.join("azure-analysis-response.json"), json).unwrap();
    }

    #[tokio::test]
    async fn test_azure_analysis_high_resolution() {
        initialize().await;
        let test_file_path = Path::new("./input/test.png");
        let output_dir = Path::new("./output/azure");
        fs::create_dir_all(output_dir).unwrap();
        let temp_file = NamedTempFile::new().unwrap();
        fs::copy(test_file_path, temp_file.path()).unwrap();
        let result = azure_analysis(
            &temp_file,
            Some(vec![DocumentAnalysisFeature::OcrHighResolution
                .as_str()
                .to_string()]),
            SegmentationStrategy::LayoutAnalysis,
        )
        .await
        .expect("Azure analysis failed");
        let json = serde_json::to_string_pretty(&result).unwrap();
        fs::write(
            output_dir.join("azure-analysis-response-high-resolution.json"),
            json,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn test_azure_analysis_page() {
        initialize().await;
        let test_file_path = Path::new("./input/test.png");
        let output_dir = Path::new("./output/azure");
        fs::create_dir_all(output_dir).unwrap();
        let temp_file = NamedTempFile::new().unwrap();
        fs::copy(test_file_path, temp_file.path()).unwrap();
        let result = azure_analysis(&temp_file, None, SegmentationStrategy::Page)
            .await
            .unwrap();
        let json = serde_json::to_string_pretty(&result).unwrap();
        fs::write(output_dir.join("azure-analysis-response-page.json"), json).unwrap();
    }
}
