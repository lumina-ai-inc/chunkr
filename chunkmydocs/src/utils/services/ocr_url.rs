use crate::models::server::segment::OCRResult;
use crate::utils::services::html::{convert_table_to_markdown, extract_table_html};
use crate::utils::services::ocr::{perform_formula_ocr, perform_general_ocr, perform_table_ocr};
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
    let ocr_results = match perform_general_ocr(original_file.path()).await {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    Ok((ocr_results, "".to_string(), "".to_string()))
}

pub async fn download_and_formula_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let latex_formula = match perform_formula_ocr(original_file.path()).await {
        Ok(latex_formula) => latex_formula,
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    Ok((
        vec![],
        format!("<span class=\"formula\">{}</span>", latex_formula.clone()),
        format!("${}$", latex_formula),
    ))
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
    let table_ocr_task =
        tokio::task::spawn(async move { perform_table_ocr(&original_file_path).await });
    let general_ocr_task =
        tokio::task::spawn(async move { perform_general_ocr(&original_file_path_clone).await });
    let ocr_results = match general_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap_or_default(),
        Err(e) => {
            println!("Error getting OCR results: {}", e);
            vec![]
        }
    };

    let table_ocr_result: Result<(String, String), Box<dyn std::error::Error>> =
        match table_ocr_task.await {
            Ok(table_result) => match table_result {
                Ok((html, markdown)) => Ok((html, markdown)),
                Err(e) => {
                    println!("Error unwrapping table OCR results: {}", e);
                    Ok(("".to_string(), "".to_string()))
                }
            },
            Err(e) => {
                println!("Error getting table OCR results: {}", e);
                Ok(("".to_string(), "".to_string()))
            }
        };

    match table_ocr_result {
        Ok(result) => Ok((ocr_results, result.0, result.1)),
        Err(e) => {
            println!("Error getting table OCR results: {}", e);
            Ok((ocr_results, "".to_string(), "".to_string()))
        }
    }
}
