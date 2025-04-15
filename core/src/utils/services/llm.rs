use crate::configs::llm_config::{Config as LlmConfig, LlmModel};
use crate::models::llm::LlmProcessing;
use crate::models::open_ai::{Message, MessageContent, OpenAiRequest, OpenAiResponse};
use crate::utils::rate_limit::{get_llm_rate_limiter, LLM_TIMEOUT, TOKEN_TIMEOUT};
use crate::utils::retry::retry_with_backoff;
use std::error::Error;
use std::fmt;

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

    if let Some(timeout) = LLM_TIMEOUT.get() {
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
            println!("Error parsing JSON: {:?}\nResponse: {}", e, text);
            return Err(Box::new(LLMError("Error parsing JSON".to_string())));
        }
    };
    Ok(response)
}

/// Process an OpenAI request with rate limiting and retrying on failure.
async fn open_ai_call_handler(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    response_format: Option<serde_json::Value>,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    let rate_limiter = get_llm_rate_limiter(&url);
    retry_with_backoff(|| async {
        let rate_limiter = rate_limiter.clone();
        if let Some(rate_limiter) = rate_limiter {
            rate_limiter
                .acquire_token_with_timeout(std::time::Duration::from_secs(
                    *TOKEN_TIMEOUT.get().unwrap(),
                ))
                .await?;
        }
        open_ai_call(
            url.clone(),
            key.clone(),
            model.clone(),
            messages.clone(),
            max_completion_tokens,
            temperature,
            response_format.clone(),
        )
        .await
    })
    .await
}

/// Process an OpenAI request
/// If the request fails, it will retry with the fallback model if provided.
async fn process_openai_request(
    model: LlmModel,
    fallback_model: Option<LlmModel>,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    response_format: Option<serde_json::Value>,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    match open_ai_call_handler(
        model.provider_url,
        model.api_key,
        model.model,
        messages.clone(),
        max_completion_tokens,
        temperature,
        response_format.clone(),
    )
    .await
    {
        Ok(response) => Ok(response),
        Err(e) => {
            if let Some(fallback_model) = fallback_model {
                Ok(open_ai_call_handler(
                    fallback_model.provider_url,
                    fallback_model.api_key,
                    fallback_model.model,
                    messages,
                    max_completion_tokens,
                    temperature,
                    response_format,
                )
                .await?)
            } else {
                Err(e)
            }
        }
    }
}

fn get_llm_content(response: OpenAiResponse) -> Result<String, Box<dyn Error + Send + Sync>> {
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

pub async fn try_extract_from_llm(
    messages: Vec<Message>,
    fence_type: Option<&str>,
    fallback_content: Option<String>,
    llm_processing: LlmProcessing,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let llm_config = LlmConfig::from_env().unwrap();
    let model = llm_config.get_model(llm_processing.model_id)?;
    let fallback_model = llm_config.get_fallback_model(llm_processing.fallback_strategy)?;

    let response = match process_openai_request(
        model,
        fallback_model,
        messages.clone(),
        llm_processing.max_completion_tokens,
        Some(llm_processing.temperature),
        None,
    )
    .await
    {
        Ok(response) => response,
        Err(e) => {
            if let Some(fallback_content) = fallback_content {
                println!("LLM API request(s) failed. Using fallback content");
                return Ok(fallback_content);
            }
            return Err(e);
        }
    };

    if !response.choices.is_empty() && response.choices[0].finish_reason != "length" {
        if let Some(content) = extract_fenced_content(&get_llm_content(response)?, fence_type) {
            return Ok(content);
        }
    } else if !response.choices.is_empty() {
        println!("Response was truncated (finish_reason: length).");
    }

    if let Some(fallback_content) = fallback_content {
        println!("Using fallback content");
        return Ok(fallback_content);
    }

    Err(Box::new(LLMError(format!(
        "No valid content found | fence_type: {}",
        fence_type.unwrap_or("")
    ))))
}
