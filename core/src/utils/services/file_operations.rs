use crate::configs::worker_config::{Config as WorkerConfig, FileUrlFormat};
use crate::utils::clients;
use crate::utils::services::pdf::count_pages;
use crate::utils::storage::services::{generate_presigned_url, upload_to_s3};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::error::Error;
use std::io::Read;
use std::process::Command;
use tempfile::NamedTempFile;
use url;
use urlencoding;

pub fn check_file_type(
    file: &NamedTempFile,
    original_file_extension: Option<String>,
) -> Result<(String, String), Box<dyn Error>> {
    let output = Command::new("file")
        .arg("--mime-type")
        .arg("-b")
        .arg(file.path().to_str().unwrap())
        .output()?;

    let mime_type = String::from_utf8(output.stdout)?.trim().to_string();
    match mime_type.as_str() {
        "application/pdf" => Ok((mime_type, "pdf".to_string())),
        "application/octet-stream" => {
            if let Some(ext) = original_file_extension {
                if ext == "pdf" {
                    println!("Detected PDF file by extension");
                    match count_pages(file) {
                        Ok(pages) => {
                            println!("Detected {} pages in PDF file", pages);
                            return Ok(("application/pdf".to_string(), "pdf".to_string()));
                        }
                        Err(e) => {
                            println!("Error counting pages in PDF file: {}", e);
                            return Err(Box::new(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                format!("Unsupported file type: {}", mime_type),
                            )));
                        }
                    }
                }
            }

            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Unsupported file type: {}", mime_type),
            )))
        }
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Ok((mime_type, "docx".to_string()))
        }
        "application/msword" => Ok((mime_type, "doc".to_string())),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => {
            Ok((mime_type, "pptx".to_string()))
        }
        "application/vnd.ms-powerpoint" => Ok((mime_type, "ppt".to_string())),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => {
            Ok((mime_type, "xlsx".to_string()))
        }
        "application/vnd.ms-excel" => Ok((mime_type, "xls".to_string())),
        "image/jpeg" | "image/jpg" => Ok((mime_type, "jpg".to_string())),
        "image/png" => Ok((mime_type, "png".to_string())),
        _ => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Unsupported file type: {}", mime_type),
        ))),
    }
}

pub fn convert_to_pdf(
    input_file: &NamedTempFile,
    original_file_extension: Option<String>,
) -> Result<NamedTempFile, Box<dyn Error>> {
    let output_dir = input_file.path().parent().unwrap();

    let (mime_type, _) = check_file_type(input_file, original_file_extension)?;

    if mime_type.starts_with("image/") {
        // Use ImageMagick for image conversion
        let output_path = output_dir.join(
            input_file
                .path()
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
                + ".pdf",
        );

        let output = Command::new("convert")
            .arg(input_file.path().to_str().unwrap())
            .arg(output_path.to_str().unwrap())
            .output()?;

        if !output.status.success() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("ImageMagick conversion failed: {:?}", output),
            )));
        }

        if output_path.exists() {
            let temp_file = NamedTempFile::new()?;
            std::fs::copy(&output_path, temp_file.path())?;
            std::fs::remove_file(output_path)?; // Clean up the temporary file
            Ok(temp_file)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Converted PDF file not found in output directory",
            )))
        }
    } else {
        // Use LibreOffice for document conversion
        let output = Command::new("libreoffice")
            .args([
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                output_dir.to_str().unwrap(),
                input_file.path().to_str().unwrap(),
            ])
            .output()?;

        if !output.status.success() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("LibreOffice conversion failed: {:?}", output),
            )));
        }

        let pdf_file_name = input_file
            .path()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string()
            + ".pdf";

        let pdf_file_path = output_dir.join(pdf_file_name);

        if pdf_file_path.exists() {
            let temp_file = NamedTempFile::new()?;
            std::fs::copy(&pdf_file_path, temp_file.path())?;
            std::fs::remove_file(&pdf_file_path)?; // Clean up the temporary file
            Ok(temp_file)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Converted PDF file not found in output directory",
            )))
        }
    }
}

pub async fn get_base64(input: String) -> Result<(Vec<u8>, Option<String>), Box<dyn Error>> {
    if input.starts_with("http://") || input.starts_with("https://") {
        let client = clients::get_reqwest_client();
        let response = client.get(&input)
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
            .header("Accept", "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
            .header("Accept-Language", "en-US,en;q=0.5")
            .header("Connection", "keep-alive")
            .header("Upgrade-Insecure-Requests", "1")
            .send()
            .await?;

        let mut filename = None;
        if let Some(content_disposition) = response.headers().get("content-disposition") {
            if let Ok(header_value) = content_disposition.to_str() {
                if header_value.contains("filename=") {
                    filename = header_value
                        .split("filename=")
                        .nth(1)
                        .map(|f| f.trim_matches(|c| c == '"' || c == '\'').to_string());
                }
            }
        }

        if filename.is_none() {
            if let Ok(url) = url::Url::parse(&input) {
                if let Some(path_segments) = url.path_segments() {
                    if let Some(last_segment) = path_segments.last() {
                        if !last_segment.is_empty() {
                            filename = Some(
                                urlencoding::decode(last_segment)
                                    .unwrap_or(std::borrow::Cow::Borrowed(last_segment))
                                    .into_owned(),
                            );
                        }
                    }
                }
            }
        }

        Ok((response.bytes().await?.to_vec(), filename))
    } else if input.starts_with("data:") && input.contains(";base64,") {
        let base64_content = input.split(";base64,").nth(1).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid base64 data URL format",
            )
        })?;

        let decoded = STANDARD.decode(base64_content)?;
        Ok((decoded, None))
    } else {
        let decoded = STANDARD.decode(&input)?;
        Ok((decoded, None))
    }
}

pub async fn get_file_url(
    temp_file: &NamedTempFile,
    s3_location: &str,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let config = WorkerConfig::from_env()?;
    let (mime_type, _) = check_file_type(temp_file, None)
        .map_err(|e| -> Box<dyn Error + Send + Sync> { e.to_string().into() })?;
    match config.file_url_format {
        FileUrlFormat::Base64 => {
            let mut buffer = Vec::new();
            let mut file = temp_file.reopen()?;
            file.read_to_end(&mut buffer)?;
            let base64_data = STANDARD.encode(&buffer);
            Ok(format!("data:{};base64,{}", mime_type, base64_data))
        }
        FileUrlFormat::Url => {
            upload_to_s3(s3_location, temp_file.path())
                .await
                .map_err(|e| -> Box<dyn Error + Send + Sync> { e.to_string().into() })?;

            let presigned_url = generate_presigned_url(s3_location, true, None, false, &mime_type)
                .await
                .map_err(|e| -> Box<dyn Error + Send + Sync> { e.to_string().into() })?;

            Ok(presigned_url)
        }
    }
}
