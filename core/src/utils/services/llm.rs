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
    response_format: Option<serde_json::Value>,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    let request = OpenAiRequest {
        model: model.clone(),
        messages,
        max_completion_tokens,
        temperature,
        response_format,
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
            println!(
                "Model: {}\nError parsing JSON: {:?}\nRaw response: {}",
                model, e, text
            );
            return Err(Box::new(LLMError("Error parsing JSON".to_string())));
        }
    };
    Ok(response)
}

/// Process an OpenAI request with rate limiting and retrying on failure.
/// If the request fails, it will retry with the fallback model if provided.
/// If the response finish_reason is "length", it will retry with increased max_completion_tokens.
pub async fn process_openai_request(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    response_format: Option<serde_json::Value>,
    use_fallback: bool,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    let rate_limiter = LLM_RATE_LIMITER.get().unwrap();

    // First attempt
    let response = match retry_with_backoff(|| async {
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
            response_format.clone(),
        )
        .await
    })
    .await
    {
        Ok(response) => response,
        Err(e) => {
            if use_fallback {
                let llm_config = LlmConfig::from_env().unwrap();
                if let Some(fallback_model) = llm_config.fallback_model {
                    println!("Using fallback model: {}", fallback_model);
                    retry_with_backoff(|| async {
                        rate_limiter
                            .acquire_token_with_timeout(std::time::Duration::from_secs(
                                *TOKEN_TIMEOUT.get().unwrap(),
                            ))
                            .await?;
                        open_ai_call(
                            url.clone(),
                            key.clone(),
                            fallback_model.clone(),
                            messages.clone(),
                            max_completion_tokens.clone(),
                            temperature.clone(),
                            response_format.clone(),
                        )
                        .await
                    })
                    .await?
                } else {
                    println!("No fallback model provided");
                    return Err(e);
                }
            } else {
                println!("Fallback not enabled");
                return Err(e);
            }
        }
    };

    // Check if the finish reason was "length" and retry with more tokens if needed
    if !response.choices.is_empty() && response.choices[0].finish_reason == "length" {
        println!("Response was truncated (finish_reason: length).");

        // Try up to 3 times
        let mut current_response = response;

        for retry_count in 1..=3 {
            rate_limiter
                .acquire_token_with_timeout(std::time::Duration::from_secs(
                    *TOKEN_TIMEOUT.get().unwrap(),
                ))
                .await?;

            println!("Attempt {} of 3", retry_count);

            current_response = open_ai_call(
                url.clone(),
                key.clone(),
                model.clone(),
                messages.clone(),
                max_completion_tokens.clone(),
                temperature.clone(),
                response_format.clone(),
            )
            .await?;

            // If we got a complete response, break out of the loop
            if current_response.choices.is_empty()
                || current_response.choices[0].finish_reason != "length"
            {
                println!("Received complete response on retry {}", retry_count);
                break;
            }

            if retry_count < 3 {
                println!("Response still truncated, continuing to retry");
            }
        }

        return Ok(current_response);
    }

    Ok(response)
}

pub fn create_basic_message(role: String, prompt: String) -> Result<Message, Box<dyn Error>> {
    Ok(Message {
        role,
        content: MessageContent::String { content: prompt },
    })
}

pub fn create_basic_image_message(
    role: String,
    prompt: String,
    temp_file: &NamedTempFile,
) -> Result<Message, Box<dyn Error>> {
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
    Ok(Message {
        role,
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
    })
}

pub async fn llm_ocr(
    temp_file: &NamedTempFile,
    prompt: String,
    temperature: Option<f32>,
    fallback_model: Option<String>,
    use_fallback: bool,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let message = create_basic_image_message("user".to_string(), prompt, temp_file)
        .map_err(|e| Box::new(LLMError(e.to_string())) as Box<dyn Error + Send + Sync>)?;
    let llm_config = LlmConfig::from_env().unwrap();

    let response = process_openai_request(
        llm_config.ocr_url.unwrap_or(llm_config.url),
        llm_config.ocr_key.unwrap_or(llm_config.key),
        fallback_model.unwrap_or(llm_config.ocr_model.unwrap_or(llm_config.model)),
        vec![message],
        None,
        temperature,
        None,
        use_fallback,
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
    fallback_content: Option<String>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let worker_config = WorkerConfig::from_env().unwrap();
    let max_retries = worker_config.max_retries;
    let use_fallback = fallback_content.is_some();
    for attempt in 0..max_retries {
        let temperature = (attempt as f32) * 0.2;
        if temperature > 1.0 {
            break;
        }
        let response = llm_ocr(
            temp_file,
            prompt.clone(),
            Some(temperature),
            None,
            use_fallback,
        )
        .await?;
        if let Some(content) = extract_fenced_content(&response, fence_type) {
            return Ok(content);
        }
    }

    if let Some(fallback_content) = fallback_content {
        println!("Using fallback content");
        return Ok(fallback_content);
    }

    Err(Box::new(LLMError(format!(
        "No {} content found after {} attempts",
        fence_type.unwrap_or(""),
        max_retries
    ))))
}

pub async fn html_ocr(
    temp_file: &NamedTempFile,
    prompt: String,
    fallback_content: Option<String>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    retry_ocr_with_temperature(temp_file, prompt, Some("html"), fallback_content).await
}

pub async fn markdown_ocr(
    temp_file: &NamedTempFile,
    prompt: String,
    fallback_content: Option<String>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    retry_ocr_with_temperature(temp_file, prompt, Some("markdown"), fallback_content).await
}

pub async fn latex_ocr(
    temp_file: &NamedTempFile,
    prompt: String,
    fallback_content: Option<String>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    retry_ocr_with_temperature(temp_file, prompt, Some("latex"), fallback_content).await
}

pub async fn llm_segment(
    temp_file: &NamedTempFile,
    prompt: String,
    fallback_content: Option<String>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    retry_ocr_with_temperature(temp_file, prompt, None, fallback_content).await
}
