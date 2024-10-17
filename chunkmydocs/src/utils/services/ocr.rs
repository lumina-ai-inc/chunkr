use crate::models::server::segment::OCRResult;
use crate::utils::services::rapid_ocr::call_rapid_ocr_api;
use crate::utils::services::table_struct::recognize_table;
use crate::utils::storage::services::download_to_tempfile;
use aws_sdk_s3::Client as S3Client;
use reqwest::Client as ReqwestClient;

pub async fn download_and_ocr(
    s3_client: &S3Client,
    reqwest_client: &ReqwestClient,
    image_location: &str,
) -> Result<Vec<OCRResult>, Box<dyn std::error::Error>> {
    println!("Downloading and OCRing image: {:?}", image_location);
    let temp_file = download_to_tempfile(s3_client, reqwest_client, image_location, None).await?;

    let ocr_results = call_rapid_ocr_api(temp_file.path()).await?;
    // let table_structures = recognize_table(binarized_temp_file.path()).await?;

    // temp_file.close()?;
    // binarized_temp_file.close()?;

    Ok(ocr_results)
}
