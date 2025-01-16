use crate::configs::llm_config::Config as LlmConfig;
use crate::configs::worker_config::Config as WorkerConfig;
use crate::models::chunkr::open_ai::{
    ContentPart, ImageUrl, Message, MessageContent, OpenAiRequest, OpenAiResponse,
};
use crate::utils::rate_limit::{LLM_OCR_TIMEOUT, LLM_RATE_LIMITER, TOKEN_TIMEOUT};
use crate::utils::retry::retry_with_backoff;
use base64::{engine::general_purpose, Engine as _};
use std::error::Error;
use std::fmt;
use std::io::Read;
use tempfile::NamedTempFile;

#[derive(Debug)]
struct LLMError(String);

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LLMError {}

pub async fn open_ai_call(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    let request = OpenAiRequest {
        model,
        messages,
        max_completion_tokens,
        temperature,
    };
    let client = reqwest::Client::new();
    let mut openai_request = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", key));

    if let Some(timeout) = LLM_OCR_TIMEOUT.get() {
        if let Some(timeout_value) = timeout {
            openai_request = openai_request.timeout(std::time::Duration::from_secs(*timeout_value));
        }
    }

    let response = openai_request
        .json(&request)
        .send()
        .await?
        .error_for_status()?;

    let text = response.text().await?;
    let response: OpenAiResponse = match serde_json::from_str(&text) {
        Ok(parsed) => parsed,
        Err(e) => {
            println!("Error parsing JSON: {:?}\nRaw response: {}", e, text);
            return Err(Box::new(LLMError("Error parsing JSON".to_string())));
        }
    };
    Ok(response)
}

/// Process an OpenAI request with rate limiting and retrying on failure
pub async fn process_openai_request(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    let rate_limiter = LLM_RATE_LIMITER.get().unwrap();
    Ok(retry_with_backoff(|| async {
        rate_limiter
            .acquire_token_with_timeout(std::time::Duration::from_secs(
                *TOKEN_TIMEOUT.get().unwrap(),
            ))
            .await?;
        open_ai_call(
            url.clone(),
            key.clone(),
            model.clone(),
            messages.clone(),
            max_completion_tokens.clone(),
            temperature.clone(),
        )
        .await
    })
    .await?)
}

pub fn get_basic_message(prompt: String) -> Result<Vec<Message>, Box<dyn Error>> {
    Ok(vec![Message {
        role: "user".to_string(),
        content: MessageContent::String { content: prompt },
    }])
}

pub fn get_basic_image_message(
    temp_file: &NamedTempFile,
    prompt: String,
) -> Result<Vec<Message>, Box<dyn Error>> {
    let mut buffer = Vec::new();
    let mut file = temp_file.reopen()?;
    file.read_to_end(&mut buffer)?;
    let base64_image = general_purpose::STANDARD.encode(&buffer);
    if base64_image.is_empty() {
        return Err(Box::new(LLMError(format!(
            "No image data (bytes read: {})",
            buffer.len()
        ))));
    }
    Ok(vec![Message {
        role: "user".to_string(),
        content: MessageContent::Array {
            content: vec![
                ContentPart {
                    content_type: "text".to_string(),
                    text: Some(prompt),
                    image_url: None,
                },
                ContentPart {
                    content_type: "image_url".to_string(),
                    text: None,
                    image_url: Some(ImageUrl {
                        url: format!("data:image/jpeg;base64,{}", base64_image),
                    }),
                },
            ],
        },
    }])
}

pub async fn llm_ocr(
    temp_file: &NamedTempFile,
    prompt: String,
    temperature: Option<f32>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let messages = get_basic_image_message(temp_file, prompt)
        .map_err(|e| Box::new(LLMError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    let llm_config = LlmConfig::from_env().unwrap();

    let response = process_openai_request(
        llm_config.ocr_url.unwrap_or(llm_config.url),
        llm_config.ocr_key.unwrap_or(llm_config.key),
        llm_config.ocr_model.unwrap_or(llm_config.model),
        messages.clone(),
        None,
        temperature,
    )
    .await?;

    if let MessageContent::String { content } = response.choices[0].message.content.clone() {
        Ok(content)
    } else {
        Err(Box::new(LLMError("Invalid content type".to_string())) as Box<dyn Error + Send + Sync>)
    }
}

fn extract_fenced_content(content: &str, fence_type: Option<&str>) -> Option<String> {
    let split_pattern = match fence_type {
        Some(ft) => format!("```{}", ft),
        None => "```".to_string(),
    };

    content
        .split(&split_pattern)
        .nth(1)
        .and_then(|content| content.split("```").next())
        .map(|content| content.trim().to_string())
}

/// Retry OCR with increasing temperature until a valid response is found
async fn retry_ocr_with_temperature(
    temp_file: &NamedTempFile,
    prompt: String,
    fence_type: Option<&str>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let worker_config = WorkerConfig::from_env().unwrap();
    let max_retries = worker_config.max_retries;
    let mut attempt = 0;
    loop {
        let temperature = (attempt as f32) * 0.2;
        if temperature > 1.0 {
            return Err(Box::new(LLMError(format!(
                "Temperature too high after {} attempts",
                attempt
            ))));
        }

        let response = llm_ocr(temp_file, prompt.clone(), Some(temperature)).await?;
        if let Some(content) = extract_fenced_content(&response, fence_type) {
            return Ok(content);
        }
        attempt += 1;
        if attempt >= max_retries {
            return Err(Box::new(LLMError(format!(
                "No {} content found after {} attempts",
                fence_type.unwrap_or(""),
                attempt
            ))));
        }
    }
}

pub async fn html_ocr(
    temp_file: &NamedTempFile,
    prompt: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    retry_ocr_with_temperature(temp_file, prompt, Some("html")).await
}

pub async fn markdown_ocr(
    temp_file: &NamedTempFile,
    prompt: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    retry_ocr_with_temperature(temp_file, prompt, Some("markdown")).await
}

pub async fn latex_ocr(
    temp_file: &NamedTempFile,
    prompt: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    retry_ocr_with_temperature(temp_file, prompt, Some("latex")).await
}

pub async fn llm_segment(
    temp_file: &NamedTempFile,
    prompt: String,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    retry_ocr_with_temperature(temp_file, prompt, None).await
}
