use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::general_ocr::DoctrResponse;
use crate::models::chunkr::output::OCRResult;
use crate::utils::rate_limit::GENERAL_OCR_TIMEOUT;
use crate::utils::retry::retry_with_backoff;
use std::error::Error;
use std::fmt;
use tempfile::NamedTempFile;

#[derive(Debug)]
struct OcrError(String);

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for OcrError {}

pub async fn doctr_ocr(
    temp_file: &NamedTempFile,
) -> Result<Vec<OCRResult>, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let worker_config = WorkerConfig::from_env()
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;

    let general_ocr_url = worker_config.general_ocr_url.unwrap();

    let url = format!("{}/ocr", &general_ocr_url);

    let file_content = tokio::fs::read(temp_file.path()).await?;

    let form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(file_content)
            .file_name(
                temp_file
                    .path()
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
            )
            .mime_str("image/jpeg")?,
    );

    let mut request = client.post(&url).multipart(form);

    if let Some(timeout) = GENERAL_OCR_TIMEOUT.get() {
        if let Some(timeout_value) = timeout {
            request = request.timeout(std::time::Duration::from_secs(*timeout_value));
        }
    }

    let response = request.send().await?.error_for_status()?;

    let doctr_response: DoctrResponse = response.json().await?;
    Ok(Vec::from(doctr_response))
}

pub async fn perform_general_ocr(
    temp_file: &NamedTempFile,
) -> Result<Vec<OCRResult>, Box<dyn Error + Send + Sync>> {
    Ok(retry_with_backoff(|| async {
        doctr_ocr(temp_file).await
    })
    .await?)
}

pub async fn perform_general_ocr_batch(
    temp_files: &[&NamedTempFile],
) -> Result<Vec<Vec<OCRResult>>, Box<dyn Error + Send + Sync>> {
    Ok(retry_with_backoff(|| async {
        doctr_ocr_batch(temp_files).await
    })
    .await?)
}

pub async fn doctr_ocr_batch(
    temp_files: &[&NamedTempFile],
) -> Result<Vec<Vec<OCRResult>>, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let worker_config = WorkerConfig::from_env()
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;

    let general_ocr_url = worker_config.general_ocr_url.unwrap();
    let url = format!("{}/batch", &general_ocr_url);

    let mut form = reqwest::multipart::Form::new();
    for temp_file in temp_files.iter() {
        let file_content = tokio::fs::read(temp_file.path()).await?;
        form = form.part(
            "files",
            reqwest::multipart::Part::bytes(file_content)
                .file_name(temp_file.path().file_name().unwrap_or_default().to_string_lossy().into_owned())
                .mime_str("image/jpeg")?,
        );
    }

    let mut request = client.post(&url).multipart(form);

    if let Some(timeout) = GENERAL_OCR_TIMEOUT.get() {
        if let Some(timeout_value) = timeout {
            request = request.timeout(std::time::Duration::from_secs(*timeout_value));
        }
    }

    let response = request.send().await?.error_for_status()?;
    let doctr_response: Vec<DoctrResponse> = response.json().await?;
    Ok(doctr_response.into_iter().map(Vec::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::throttle_config::Config as ThrottleConfig;
    use std::path::Path;

    #[tokio::test]
    async fn test_doctr_ocr() -> Result<(), Box<dyn Error + Send + Sync>> {
        let temp_file = NamedTempFile::new()?;
        std::fs::copy("input/test.jpg", temp_file.path())?;
        doctr_ocr(&temp_file).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_general_ocr() -> Result<(), Box<dyn Error + Send + Sync>> {
        let input_dir = Path::new("input");
        let first_image = std::fs::read_dir(input_dir)?
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    let path = e.path();
                    if let Some(ext) = path.extension() {
                        if ext == "jpg" || ext == "jpeg" || ext == "png" {
                            Some(path)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            })
            .next()
            .ok_or("No image files found in input directory")?;

        let mut tasks = Vec::new();
        let count = 1000;
        for _ in 0..count {
            let input_file = first_image.clone();
            let temp_file = NamedTempFile::new()?;
            std::fs::copy(&input_file, temp_file.path())?;
            let task = tokio::spawn(async move {
                match perform_general_ocr(&temp_file).await {
                    Ok(_) => Ok(()),
                    Err(e) => {
                        println!("Error processing {:?}: {:?}", input_file, e);
                        Err(e)
                    }
                }
            });
            tasks.push(task);
        }

        let start = std::time::Instant::now();
        let mut error_count = 0;
        for task in tasks {
            if let Err(e) = task.await? {
                println!("Error processing: {:?}", e);
                error_count += 1;
            }
        }
        let duration = start.elapsed();
        let images_per_second = count as f64 / duration.as_secs_f64();
        let throttle_config = ThrottleConfig::from_env().unwrap();
        println!(
            "General OCR rate limit: {:?}",
            throttle_config.general_ocr_rate_limit
        );
        println!("Time taken: {:?}", duration);
        println!("Images per second: {:?}", images_per_second);
        println!("Error count: {:?}", error_count);

        if error_count > 0 {
            Err(format!("Error count {} > 0", error_count).into())
        } else {
            Ok(())
        }
    }
}
