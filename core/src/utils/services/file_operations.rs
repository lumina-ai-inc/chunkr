use crate::utils::clients;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::error::Error;
use std::process::Command;
use tempfile::NamedTempFile;
use url;
use urlencoding;

pub fn check_file_type(file: &NamedTempFile) -> Result<(String, String), Box<dyn Error>> {
    let output = Command::new("file")
        .arg("--mime-type")
        .arg("-b")
        .arg(file.path().to_str().unwrap())
        .output()?;

    let mime_type = String::from_utf8(output.stdout)?.trim().to_string();
    println!("mime_type: {:?}", mime_type);
    match mime_type.as_str() {
        "application/pdf" => Ok((mime_type, "pdf".to_string())),
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

pub fn convert_to_pdf(input_file: &NamedTempFile) -> Result<NamedTempFile, Box<dyn Error>> {
    let output_dir = input_file.path().parent().unwrap();

    let (mime_type, _) = check_file_type(input_file)?;

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
    } else {
        let decoded = STANDARD.decode(&input)?;
        Ok((decoded, None))
    }
}
