use crate::configs::{
    feature_config::Config as FeatureConfig,
    worker_config::{Config as WorkerConfig, FileUrlFormat},
};
use crate::models::file_operations::{HtmlConversionResult, ImageConversionResult};
use crate::utils::clients;
use crate::utils::services::pdf::count_pages;
use crate::utils::storage::services::{generate_presigned_url, upload_to_s3};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::error::Error;
use std::io::Read;
use std::process::Command;
use std::sync::Arc;
use tempfile::{Builder, NamedTempFile};
use url;
use urlencoding;

pub fn check_is_spreadsheet(mime_type: &str) -> Result<bool, Box<dyn Error>> {
    if !FeatureConfig::from_env()?.enable_excel_parse {
        return Ok(false);
    }
    Ok(
        mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            || mime_type == "application/vnd.ms-excel",
    )
}

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
                            println!("Detected {pages} pages in PDF file");
                            return Ok(("application/pdf".to_string(), "pdf".to_string()));
                        }
                        Err(e) => {
                            println!("Error counting pages in PDF file: {e}");
                            return Err(Box::new(std::io::Error::other(format!(
                                "Unsupported file type: {mime_type}"
                            ))));
                        }
                    }
                }
            }

            Err(Box::new(std::io::Error::other(format!(
                "Unsupported file type: {mime_type}"
            ))))
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
        _ => Err(Box::new(std::io::Error::other(format!(
            "Unsupported file type: {mime_type}"
        )))),
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
            return Err(Box::new(std::io::Error::other(format!(
                "ImageMagick conversion failed: {output:?}"
            ))));
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
            return Err(Box::new(std::io::Error::other(format!(
                "LibreOffice conversion failed: {output:?}"
            ))));
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
                if let Some(mut path_segments) = url.path_segments() {
                    if let Some(last_segment) = path_segments.next_back() {
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
            Ok(format!("data:{mime_type};base64,{base64_data}"))
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

pub fn convert_to_html(input_file: &NamedTempFile) -> Result<HtmlConversionResult, Box<dyn Error>> {
    let temp_dir = tempfile::tempdir()?;
    let output_dir = temp_dir.path();

    // Use LibreOffice for document conversion
    let output = Command::new("libreoffice")
        .args([
            "--headless",
            "--convert-to",
            "html",
            "--outdir",
            output_dir.to_str().unwrap(),
            input_file.path().to_str().unwrap(),
        ])
        .output()?;

    if !output.status.success() {
        return Err(Box::new(std::io::Error::other(format!(
            "LibreOffice conversion failed: {output:?}"
        ))));
    }

    let html_file = Builder::new().suffix(".html").tempfile()?;
    let mut embedded_images: Vec<ImageConversionResult> = Vec::new();
    let mut html_content = String::new();

    for entry in output_dir.read_dir()? {
        let entry = entry?;
        if entry.path().is_file() {
            match entry.path().extension().and_then(|ext| ext.to_str()) {
                Some("html") => {
                    html_content = std::fs::read_to_string(entry.path())?;
                }
                Some("png") => {
                    let img_file = NamedTempFile::new()?;
                    std::fs::copy(entry.path(), img_file.path())?;

                    // Get the file name
                    if let Some(file_name) = entry.path().file_name().and_then(|name| name.to_str())
                    {
                        embedded_images.push(ImageConversionResult::new(
                            Arc::new(img_file),
                            file_name.to_string(),
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    for image in embedded_images.iter_mut() {
        let image_name = image.image_file.path().display().to_string();
        html_content = html_content.replace(&image.html_reference, &image_name);
        image.html_reference = image_name;
    }

    // Write the updated HTML content to the final file
    std::fs::write(html_file.path(), html_content)?;

    Ok(HtmlConversionResult::new(
        Arc::new(html_file),
        embedded_images,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_convert_to_html() {
        // Create a test input file path - adjust this path as needed
        let mut input_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_file_path.push("input/test.xlsx");

        // Skip test if input file doesn't exist
        if !input_file_path.exists() {
            println!("Test file not found at {input_file_path:?}, skipping test");
            return;
        }

        let input_file = NamedTempFile::new().unwrap();
        fs::copy(input_file_path.clone(), input_file.path()).unwrap();

        // Call convert_to_html
        let result = convert_to_html(&input_file).unwrap();

        // Save the output for inspection
        let output_dir = PathBuf::from("output/file_operations/html_conversion");
        fs::create_dir_all(&output_dir).unwrap();

        // Read the HTML content and process image references
        let mut html_content = fs::read_to_string(result.html_file.path()).unwrap();

        result
            .embedded_images
            .iter()
            .enumerate()
            .for_each(|(i, img)| {
                let filename = format!("image_{}_{}.png", i, img.image_id);
                fs::copy(img.image_file.path(), output_dir.join(&filename)).unwrap();
                println!("Saved image: {} -> {}", img.html_reference, filename);
                html_content = html_content.replace(&img.html_reference, &filename);
                println!("Replaced image: {} -> {}", img.html_reference, filename);
            });

        // Save the updated HTML file with corrected image references
        fs::write(output_dir.join("test_output.html"), html_content.clone()).unwrap();

        println!("Conversion completed successfully!");
        println!(
            "HTML file saved to: {:?}",
            output_dir.join("test_output.html")
        );
        println!("Images saved to: {output_dir:?}");
        println!("Total embedded images: {}", result.embedded_images.len());

        // Print image file sizes for quality assessment
        result.embedded_images.iter().for_each(|img| {
            if let Ok(metadata) = fs::metadata(img.image_file.path()) {
                println!(
                    "Image {} size: {} bytes",
                    img.html_reference,
                    metadata.len()
                );
            }
        });
    }
}
