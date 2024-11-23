use crate::models::server::segment::{OCRResult, SegmentType};
use crate::utils::services::ocr::{
    perform_formula_ocr, perform_general_ocr, perform_page_ocr, perform_table_ocr,
};
use crate::utils::storage::services::download_to_tempfile;
use aws_sdk_s3::Client as S3Client;
use reqwest::Client as ReqwestClient;

pub async fn ocr_segments(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
    segment_type: SegmentType,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    match segment_type {
        SegmentType::Formula => {
            download_and_formula_ocr(s3_client, reqwest_client, image_location).await
        }
        SegmentType::Table => {
            download_and_table_ocr(s3_client, reqwest_client, image_location).await
        }
        SegmentType::Page => {
            download_and_ocr_page(s3_client, reqwest_client, image_location).await
        }
        _ => download_and_ocr(s3_client, reqwest_client, image_location).await,
    }
}

async fn download_and_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let general_ocr_results = match perform_general_ocr(original_file.path()).await {
        Ok(ocr_results) => ocr_results,
        Err(e) => {
            return Err(e.to_string().into());
        }
    };
    Ok((general_ocr_results, "".to_string(), "".to_string()))
}

async fn download_and_formula_ocr(
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

async fn download_and_table_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let original_file_path = original_file.path().to_owned();
    let original_file_path_clone = original_file_path.clone();
    let general_ocr_task =
        tokio::task::spawn(async move { perform_general_ocr(&original_file_path_clone).await });
    let table_ocr_task =
        tokio::task::spawn(async move { perform_table_ocr(&original_file_path).await });

    let ocr_results = match general_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap_or_default(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    let table_ocr_result = match table_ocr_task.await {
        Ok(table_result) => table_result.unwrap_or_default(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    Ok((ocr_results, table_ocr_result.0, table_ocr_result.1))
}
async fn download_and_ocr_page(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<(Vec<OCRResult>, String, String), Box<dyn std::error::Error>> {
    println!("downloading and ocr page");
    let original_file =
        download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;
    let original_file_path = original_file.path().to_owned();
    let original_file_path_clone = original_file_path.clone();
    let general_ocr_task =
        tokio::task::spawn(async move { perform_general_ocr(&original_file_path_clone).await });
    let page_ocr_task =
        tokio::task::spawn(async move { perform_page_ocr(&original_file_path).await });

    let ocr_results = match general_ocr_task.await {
        Ok(ocr_results) => ocr_results.unwrap_or_default(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    let page_ocr_result = match page_ocr_task.await {
        Ok(page_result) => page_result.unwrap_or_default(),
        Err(e) => {
            return Err(e.to_string().into());
        }
    };

    Ok((ocr_results, page_ocr_result.0, page_ocr_result.1))
}
