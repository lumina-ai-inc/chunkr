use crate::{
    models::chunkr::general_ocr::DoctrResponse,
    models::chunkr::open_ai::MessageContent,
    models::chunkr::output::OCRResult,
    utils::configs::llm_config::{get_prompt, Config as LlmConfig},
    utils::configs::worker_config::Config as WorkerConfig,
    utils::db::deadpool_redis::{create_pool as create_redis_pool, Pool},
    utils::rate_limit::{create_general_ocr_rate_limiter, create_llm_rate_limiter, RateLimiter},
    utils::services::html::clean_img_tags,
    utils::services::llm::{get_basic_image_message, open_ai_call},
};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use tempfile::NamedTempFile;

static GENERAL_OCR_RATE_LIMITER: OnceCell<RateLimiter> = OnceCell::new();
static LLM_RATE_LIMITER: OnceCell<RateLimiter> = OnceCell::new();
static POOL: OnceCell<Pool> = OnceCell::new();
static GENERAL_OCR_TIMEOUT: OnceCell<u64> = OnceCell::new();
static TOKEN_TIMEOUT: OnceCell<u64> = OnceCell::new();

fn init_throttle() {
    let llm_config = LlmConfig::from_env().unwrap();
    let llm_ocr_url = llm_config.ocr_url.unwrap_or(llm_config.url);
    let domain_name = llm_ocr_url
        .split("://")
        .nth(1)
        .ok_or("Invalid URL format: missing protocol separator")
        .and_then(|s| {
            s.split('/')
                .next()
                .ok_or("Invalid URL format: missing domain")
        })
        .unwrap_or_else(|_| "localhost");
    POOL.get_or_init(|| create_redis_pool());
    GENERAL_OCR_RATE_LIMITER.get_or_init(|| {
        create_general_ocr_rate_limiter(POOL.get().unwrap().clone(), "general_ocr")
    });
    LLM_RATE_LIMITER
        .get_or_init(|| create_llm_rate_limiter(POOL.get().unwrap().clone(), domain_name));
    GENERAL_OCR_TIMEOUT.get_or_init(|| 120);
    TOKEN_TIMEOUT.get_or_init(|| 10000);
}

#[derive(Debug)]
struct OcrError(String);

impl fmt::Display for OcrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for OcrError {}

pub async fn doctr_ocr(
    temp_file: Arc<NamedTempFile>,
) -> Result<Vec<OCRResult>, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let worker_config = WorkerConfig::from_env()
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;

    let general_ocr_url = worker_config
        .general_ocr_url
        .ok_or_else(|| "General OCR URL is not set in config".to_string())?;

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

    let response = client
        .post(&url)
        .multipart(form)
        .timeout(std::time::Duration::from_secs(
            *GENERAL_OCR_TIMEOUT.get().unwrap(),
        ))
        .send()
        .await?
        .error_for_status()?;

    let doctr_response: DoctrResponse = response.json().await?;
    Ok(Vec::from(doctr_response))
}

pub async fn llm_ocr(
    file_path: &Path,
    prompt: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let llm_config = LlmConfig::from_env().unwrap();
    let messages = get_basic_image_message(file_path, prompt)
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    let response = open_ai_call(
        llm_config.ocr_url.unwrap_or(llm_config.url),
        llm_config.ocr_key.unwrap_or(llm_config.key),
        llm_config.ocr_model.unwrap_or(llm_config.model),
        messages,
        None,
        Some(0.0),
    )
    .await
    .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    if let MessageContent::String { content } = response.choices[0].message.content.clone() {
        Ok(content)
    } else {
        Err("Invalid content type".into())
    }
}

fn extract_fenced_content(content: &str, fence_type: &str) -> Option<String> {
    content
        .split(&format!("```{}", fence_type))
        .nth(1)
        .and_then(|content| content.split("```").next())
        .map(|content| content.trim().to_string())
}

pub fn get_html_from_llm_table_ocr(
    table_ocr_result: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let html = extract_fenced_content(&table_ocr_result, "html")
        .ok_or_else(|| "No HTML content found in table OCR result")?;
    let cleaned_html = clean_img_tags(&html);
    Ok(cleaned_html)
}

pub fn get_markdown_from_llm_table_ocr(
    table_ocr_result: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    extract_fenced_content(&table_ocr_result, "markdown")
        .ok_or_else(|| "No markdown content found in table OCR result".into())
}

pub fn get_html_from_llm_page_ocr(
    page_ocr_result: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let html = extract_fenced_content(&page_ocr_result, "html")
        .ok_or_else(|| "No HTML content found in page OCR result")?;
    let cleaned_html = clean_img_tags(&html);
    Ok(cleaned_html)
}

pub fn get_markdown_from_llm_page_ocr(
    page_ocr_result: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    extract_fenced_content(&page_ocr_result, "markdown")
        .ok_or_else(|| "No markdown content found in page OCR result".into())
}

pub fn get_latex_from_llm_formula_ocr(
    formula_ocr_result: String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    extract_fenced_content(&formula_ocr_result, "latex")
        .ok_or_else(|| "No LaTeX content found in formula OCR result".into())
}

pub async fn perform_general_ocr(
    temp_file: Arc<NamedTempFile>,
) -> Result<Vec<OCRResult>, Box<dyn Error + Send + Sync>> {
    init_throttle();
    let rate_limiter = GENERAL_OCR_RATE_LIMITER.get().unwrap();
    rate_limiter
        .acquire_token_with_timeout(std::time::Duration::from_secs(
            *TOKEN_TIMEOUT.get().unwrap(),
        ))
        .await?;
    doctr_ocr(temp_file).await
}

pub async fn perform_table_ocr(
    file_path: &Path,
) -> Result<(String, String), Box<dyn Error + Send + Sync>> {
    init_throttle();
    let rate_limiter = LLM_RATE_LIMITER.get().unwrap();
    rate_limiter
        .acquire_token_with_timeout(std::time::Duration::from_secs(
            *TOKEN_TIMEOUT.get().unwrap(),
        ))
        .await?;
    let html_prompt = get_prompt("html_table", &HashMap::new())?;
    let md_prompt = get_prompt("md_table", &HashMap::new())?;
    let html_task = llm_ocr(file_path, html_prompt);
    let markdown_task = llm_ocr(file_path, md_prompt);
    let (html_response, markdown_response) = tokio::try_join!(html_task, markdown_task)?;
    let html = get_html_from_llm_table_ocr(html_response)
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    let markdown = get_markdown_from_llm_table_ocr(markdown_response)
        .map_err(|e| Box::new(OcrError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    Ok((html, markdown))
}

pub async fn perform_page_ocr(
    file_path: &Path,
) -> Result<(String, String), Box<dyn Error + Send + Sync>> {
    init_throttle();
    let rate_limiter = LLM_RATE_LIMITER.get().unwrap();
    rate_limiter
        .acquire_token_with_timeout(std::time::Duration::from_secs(
            *TOKEN_TIMEOUT.get().unwrap(),
        ))
        .await?;
    let html_prompt = get_prompt("html_page", &HashMap::new())?;
    let md_prompt = get_prompt("md_page", &HashMap::new())?;
    let html_task = llm_ocr(file_path, html_prompt);
    let markdown_task = llm_ocr(file_path, md_prompt);
    let (html_response, markdown_response) = tokio::try_join!(html_task, markdown_task)?;
    let html = get_html_from_llm_page_ocr(html_response)?;
    let markdown = get_markdown_from_llm_page_ocr(markdown_response)?;
    Ok((html, markdown))
}

pub async fn perform_formula_ocr(file_path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
    init_throttle();
    let rate_limiter = LLM_RATE_LIMITER.get().unwrap();
    rate_limiter
        .acquire_token_with_timeout(std::time::Duration::from_secs(
            *TOKEN_TIMEOUT.get().unwrap(),
        ))
        .await?;
    let prompt = get_prompt("formula", &HashMap::new())?;
    let latex_formula = llm_ocr(file_path, prompt).await?;
    get_latex_from_llm_formula_ocr(latex_formula)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::configs::throttle_config::Config as ThrottleConfig;

    #[tokio::test]
    async fn test_doctr_ocr() -> Result<(), Box<dyn Error + Send + Sync>> {
        let temp_file = NamedTempFile::new()?;
        std::fs::copy("input/test.jpg", temp_file.path())?;
        doctr_ocr(Arc::new(temp_file)).await?;
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
                match perform_general_ocr(Arc::new(temp_file)).await {
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

    #[test]
    fn test_valid_html() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td>1</td>
                <td>2</td>
            </tr>
        </table>
        ```
        "#;
        let html = get_html_from_llm_table_ocr(input.to_string()).unwrap();
        assert_eq!(
            html,
            r#"<table>
            <tr>
                <td>1</td>
                <td>2</td>
            </tr>
        </table>"#
        );
    }

    #[test]
    fn test_invalid_html() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td>1
                <td>2</td>
            </tr>
        </table>
        ```
        "#;
        let result = get_html_from_llm_table_ocr(input.to_string());
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Mismatched HTML tags: expected </td>, found </tr>"
        );
    }

    #[test]
    fn test_image_alt_text() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td>1</td>
                <td>2</td>
                <td><img src="pizza.png" alt="pizza"></td>
            </tr>
        </table>
        "#;
        let html = get_html_from_llm_table_ocr(input.to_string()).unwrap();
        assert_eq!(
            html,
            r#"<table>
            <tr>
                <td>1</td>
                <td>2</td>
                <td>pizza</td>
            </tr>
        </table>"#
        );
    }

    #[test]
    fn test_image_no_alt_text() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td>1</td>
                <td>2</td>
                <td><img src="pizza.png"></td>
            </tr>
        </table>
        "#;
        let html = get_html_from_llm_table_ocr(input.to_string()).unwrap();
        assert_eq!(
            html,
            r#"<table>
            <tr>
                <td>1</td>
                <td>2</td>
                <td></td>
            </tr>
        </table>"#
        );
    }

    #[test]
    fn test_void_elements() {
        let input = r#"
        ```html
        <table>
            <tr>
                <td><img src="pizza.png" /></td>
                <td>2<br>3</td>
            </tr>
        </table>
        ```
        "#;
        let html = get_html_from_llm_table_ocr(input.to_string()).unwrap();
        assert_eq!(
            html,
            r#"<table>
            <tr>
                <td></td>
                <td>2<br>3</td>
            </tr>
        </table>"#
        );
    }
}
