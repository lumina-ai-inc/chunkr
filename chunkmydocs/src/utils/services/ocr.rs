use crate::models::server::segment::OCRResult;
use crate::utils::services::rapid_ocr::call_rapid_ocr_api;
use crate::utils::services::table_ocr::paddle_table_ocr;
use crate::utils::storage::services::download_to_tempfile;
use aws_sdk_s3::Client as S3Client;
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client as ReqwestClient;

static TABLE_CONTENT_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)<table[^>]*>(.*?)<\/table>").unwrap()
});

pub async fn download_and_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file = download_to_tempfile(
        s3_client,
        reqwest_client,
        image_location,
        None
    ).await?;
    let ocr_results = match call_rapid_ocr_api(&original_file.path()).await {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    Ok((ocr_results, "".to_string(), "".to_string()))
}

pub async fn download_and_table_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file = download_to_tempfile(
        s3_client,
        reqwest_client,
        image_location,
        None
    ).await?;
    let original_file_path = original_file.path().to_owned();
    let original_file_path_clone = original_file_path.clone();
    let table_structure_task = tokio::task::spawn(async move {
        paddle_table_ocr(&original_file_path).await
    });
    let rapid_ocr_task = tokio::task::spawn(async move {
        call_rapid_ocr_api(&original_file_path_clone).await
    });
    let ocr_results = match rapid_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap_or_default(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    let table_structure = match table_structure_task.await {
        Ok(table_structures) => table_structures.unwrap_or_default().first().unwrap().clone(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    let html = extract_table_html(table_structure.html.clone());
    let markdown = get_table_markdown(html.clone());
    Ok((ocr_results, html, markdown))
}

fn extract_table_html(html: String) -> String {
    let mut contents = Vec::new();
    for cap in TABLE_CONTENT_REGEX.captures_iter(&html) {
        if let Some(content) = cap.get(1) {
            contents.push(content.as_str().to_string());
        }
    }
    contents.first().unwrap().to_string()
}

fn get_table_markdown(html: String) -> String {
    let markdown = html;
    return markdown;
}
