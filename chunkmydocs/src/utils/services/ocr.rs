use crate::models::server::segment::OCRResult;
use crate::utils::services::general_ocr::paddle_ocr;
use crate::utils::services::html::{convert_table_to_markdown, extract_table_html};
use crate::utils::services::table_ocr::paddle_table_ocr;
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
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let original_file_path = original_file.path().to_owned();
    let original_file_path_clone = original_file_path.clone();
    let table_structure_task =
        tokio::task::spawn(async move { paddle_table_ocr(&original_file_path).await });
    let rapid_ocr_task =
        tokio::task::spawn(async move { paddle_ocr(&original_file_path_clone).await });
    let ocr_results = match rapid_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap_or_default(),
        Err(e) => {
            println!("Error getting OCR results: {}", e);
            vec![]
        }
    };

    let table_result = match table_structure_task.await {
        Ok(table_structure) => match table_structure {
            Ok(table_structure) => {
                let first_table = table_structure.tables.first().cloned();
                match first_table {
                    Some(table) => {
                        let html = extract_table_html(table.html.clone());
                        let markdown = convert_table_to_markdown(html.clone());
                        Ok((html, markdown))
                    }
                    None => Err("No table structure found".to_string()),
                }
            }
            Err(e) => Err(e.to_string().into()),
        },
        Err(e) => Err(e.to_string()),
    };

    match table_result {
        Ok(result) => Ok((ocr_results, result.0, result.1)),
        Err(e) => {
            println!("Error getting table OCR results: {}", e);
            Ok((ocr_results, "".to_string(), "".to_string()))
        }
    }
}
