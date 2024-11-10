use crate::models::server::segment::OCRResult;
use crate::models::workers::table_ocr::PaddleTableRecognitionResult;
use crate::utils::configs::worker_config::{Config as WorkerConfig, TableOcrModel};
use crate::utils::services::general_ocr::paddle_ocr;
use crate::utils::services::html::{convert_table_to_markdown, extract_table_html};
use crate::utils::services::table_ocr::{paddle_table_ocr, vllm_table_ocr};
use crate::utils::storage::services::download_to_tempfile;
use aws_sdk_s3::Client as S3Client;

use reqwest::Client as ReqwestClient;

pub async fn download_and_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let ocr_results = match paddle_ocr(original_file.path()).await {
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
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let worker_config = WorkerConfig::from_env()?;
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let original_file_path = original_file.path().to_owned();
    let original_file_path_clone = original_file_path.clone();
    let table_ocr_task = tokio::task::spawn(async move {
        match worker_config.table_ocr_model {
            TableOcrModel::Paddle => {
                let result = paddle_table_ocr(&original_file_path).await?;
                get_html_from_paddle_table_ocr(result)
            }
            TableOcrModel::VLLM => {
                let result = vllm_table_ocr(&original_file_path).await?;
                get_html_from_vllm_table_ocr(result)
            }
        }
    });
    let paddle_ocr_task =
        tokio::task::spawn(async move { paddle_ocr(&original_file_path_clone).await });
    let ocr_results = match paddle_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap_or_default(),
        Err(e) => {
            println!("Error getting OCR results: {}", e);
            vec![]
        }
    };

    let table_ocr_result: Result<(String, String), Box<dyn std::error::Error>> =
        match table_ocr_task.await {
            Ok(html) => match html {
                Ok(html) => {
                    let html = extract_table_html(html);
                    let markdown = convert_table_to_markdown(html.clone());
                    Ok((html, markdown))
                }
                Err(e) => {
                    println!("Error getting table OCR results: {}", e);
                    Ok(("".to_string(), "".to_string()))
                }
            },
            Err(e) => Err(e.to_string().into()),
        };

    match table_ocr_result {
        Ok(result) => Ok((ocr_results, result.0, result.1)),
        Err(e) => {
            println!("Error getting table OCR results: {}", e);
            Ok((ocr_results, "".to_string(), "".to_string()))
        }
    }
}

fn get_html_from_paddle_table_ocr(
    table_ocr_result: PaddleTableRecognitionResult,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let first_table = table_ocr_result.tables.first().cloned();
    match first_table {
        Some(table) => Ok(table.html),
        None => Err("No table structure found".to_string().into()),
    }
}

// TODO: Add check for valid html
fn get_html_from_vllm_table_ocr(
    table_ocr_result: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(html_content) = table_ocr_result.split("```html").nth(1) {
        if let Some(html) = html_content.split("```").next() {
            return Ok(html.trim().to_string());
        }
    }
    Err("No HTML content found in table OCR result".into())
}
