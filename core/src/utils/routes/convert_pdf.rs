use crate::models::pdf_conversion::{PdfConversionRequest, PdfConversionResponse};
use crate::utils::services::file_operations::get_base64;
use crate::utils::services::pdf::pages_as_images;
use crate::utils::storage::services::{generate_presigned_url, upload_to_s3};
use actix_web::{web, Error, HttpResponse};
use futures::future::try_join_all;
use std::io::Write;
use tempfile::NamedTempFile;
use uuid::Uuid;

pub async fn convert_pdf(request: web::Json<PdfConversionRequest>) -> Result<HttpResponse, Error> {
    let task_id = Uuid::new_v4();

    let mut temp_pdf = NamedTempFile::new().map_err(|e| {
        actix_web::error::ErrorInternalServerError(format!("Failed to create temp file: {e}"))
    })?;

    let (pdf_bytes, _) = get_base64(request.file.clone()).await.map_err(|e| {
        actix_web::error::ErrorBadRequest(format!("Failed to download PDF from URL: {e}"))
    })?;

    temp_pdf.write_all(&pdf_bytes).map_err(|e| {
        actix_web::error::ErrorInternalServerError(format!(
            "Failed to write PDF content to temp file: {e}"
        ))
    })?;

    let image_files = pages_as_images(&temp_pdf, request.scaling_factor).map_err(|e| {
        actix_web::error::ErrorInternalServerError(format!("Failed to convert PDF to images: {e}"))
    })?;

    let base64_urls = request.base64_urls;

    // Process images concurrently
    let upload_tasks: Vec<_> = image_files
        .iter()
        .enumerate()
        .map(|(index, image_file)| {
            let page_number = index + 1;
            let s3_path = format!("s3://chunkr/pdf-conversion/{task_id}/page_{page_number}.png",);
            let image_path = image_file.path().to_path_buf();

            async move {
                upload_to_s3(&s3_path, &image_path).await.map_err(|e| {
                    actix_web::error::ErrorInternalServerError(format!(
                        "Failed to upload image to S3: {e}"
                    ))
                })?;

                let presigned_url =
                    generate_presigned_url(&s3_path, true, None, base64_urls, "image/png")
                        .await
                        .map_err(|e| {
                            actix_web::error::ErrorInternalServerError(format!(
                                "Failed to generate presigned URL: {e}"
                            ))
                        })?;

                Ok::<String, Error>(presigned_url)
            }
        })
        .collect();

    let images = try_join_all(upload_tasks).await?;

    let response = PdfConversionResponse { images };

    Ok(HttpResponse::Ok().json(response))
}
