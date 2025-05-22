use crate::configs::llm_config::{Config as LlmConfig, LlmModel};
use crate::configs::otel_config;
use crate::models::llm::LlmProcessing;
use crate::models::open_ai::{Message, MessageContent, OpenAiRequest, OpenAiResponse};
use crate::utils::rate_limit::{get_llm_rate_limiter, LLM_TIMEOUT, TOKEN_TIMEOUT};
use crate::utils::retry::retry_with_backoff;
use opentelemetry::baggage::BaggageExt;
use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::Context;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Error)]
enum LLMError {
    #[error("Error parsing JSON: {error}")]
    JsonParseError { error: String, response: String },

    #[error("{0}")]
    Generic(String),
}

pub async fn open_ai_call(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    response_format: Option<serde_json::Value>,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    println!("OpenAI call with model: {:?}", model);

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

    if let Some(Some(timeout_value)) = LLM_TIMEOUT.get() {
        openai_request = openai_request.timeout(std::time::Duration::from_secs(*timeout_value));
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
            return Err(Box::new(LLMError::JsonParseError {
                error: e.to_string(),
                response: text.trim().to_string(),
            }));
        }
    };
    Ok(response)
}

/// Process an OpenAI request with rate limiting and retrying on failure.
async fn open_ai_call_handler(
    model: LlmModel,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    response_format: Option<serde_json::Value>,
    tracer: &opentelemetry::global::BoxedTracer,
    parent_context: &Context,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    let rate_limiter = get_llm_rate_limiter(&model.id)?;

    retry_with_backoff(|| async {
        let mut span = tracer.start_with_context("open_ai_call", parent_context);
        span.set_attribute(opentelemetry::KeyValue::new("model", model.model.clone()));
        span.set_attribute(opentelemetry::KeyValue::new(
            "provider_url",
            model.provider_url.clone(),
        ));

        if let Some(segment_id) = parent_context.baggage().get("segment_id") {
            span.set_attribute(opentelemetry::KeyValue::new(
                "segment_id",
                segment_id.to_string(),
            ));
        }
        if let Some(segment_type) = parent_context.baggage().get("segment_type") {
            span.set_attribute(opentelemetry::KeyValue::new(
                "segment_type",
                segment_type.to_string(),
            ));
        }

        let ctx = parent_context.with_span(span);

        let rate_limiter = rate_limiter.clone();
        if let Some(rate_limiter) = rate_limiter {
            let model = LlmConfig::from_env()?.get_model_by_id(&model.id)?;
            ctx.span()
                .set_attribute(opentelemetry::KeyValue::new("rate_limited", true));
            if let Some(rate_limit) = model.rate_limit {
                ctx.span().set_attribute(opentelemetry::KeyValue::new(
                    "rate_limit",
                    rate_limit as f64,
                ));
            }
            rate_limiter
                .acquire_token_with_timeout(std::time::Duration::from_secs(
                    *TOKEN_TIMEOUT.get().unwrap(),
                ))
                .await?;
        } else {
            ctx.span()
                .set_attribute(opentelemetry::KeyValue::new("rate_limited", false));
        }

        match open_ai_call(
            model.provider_url.clone(),
            model.api_key.clone(),
            model.model.clone(),
            messages.clone(),
            max_completion_tokens,
            temperature,
            response_format.clone(),
        )
        .await
        {
            Ok(response) => Ok(response),
            Err(e) => {
                if let LLMError::JsonParseError { response, .. } =
                    e.downcast_ref::<LLMError>().unwrap()
                {
                    let attributes = otel_config::extract_llm_error_attributes(response);
                    for attr in attributes {
                        ctx.span().set_attribute(attr);
                    }
                }

                ctx.span()
                    .set_status(opentelemetry::trace::Status::error(e.to_string()));
                ctx.span().record_error(e.as_ref());
                ctx.span()
                    .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
                println!("Error: {}", e);
                Err(e)
            }
        }
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
    fence_type: Option<&str>,
    tracer: &opentelemetry::global::BoxedTracer,
    parent_context: &Context,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut span = tracer.start_with_context("process_openai_request", parent_context);
    span.set_attribute(opentelemetry::KeyValue::new("model", model.model.clone()));

    if let Some(segment_id) = parent_context.baggage().get("segment_id") {
        span.set_attribute(opentelemetry::KeyValue::new(
            "segment_id",
            segment_id.to_string(),
        ));
    }
    if let Some(segment_type) = parent_context.baggage().get("segment_type") {
        span.set_attribute(opentelemetry::KeyValue::new(
            "segment_type",
            segment_type.to_string(),
        ));
    }

    let ctx = parent_context.with_span(span);

    match open_ai_call_handler(
        model.clone(),
        messages.clone(),
        max_completion_tokens,
        temperature,
        response_format.clone(),
        tracer,
        &ctx,
    )
    .await
    .and_then(|response| try_extract_from_response(&response, fence_type))
    {
        Ok(response) => Ok(response),
        Err(e) => {
            if let Some(fallback_model) = fallback_model {
                ctx.span()
                    .set_attribute(opentelemetry::KeyValue::new("using_fallback", true));
                ctx.span().set_attribute(opentelemetry::KeyValue::new(
                    "fallback_model",
                    fallback_model.model.clone(),
                ));

                let response = open_ai_call_handler(
                    fallback_model.clone(),
                    messages,
                    max_completion_tokens,
                    temperature,
                    response_format,
                    tracer,
                    &ctx,
                )
                .await
                .and_then(|response| try_extract_from_response(&response, fence_type))?;
                Ok(response)
            } else {
                Err(e)
            }
        }
    }
    .inspect_err(|e| {
        ctx.span()
            .set_status(opentelemetry::trace::Status::error(e.to_string()));
        ctx.span().record_error(e.as_ref());
        ctx.span()
            .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
    })
}

/// Get the content from an OpenAI response
fn get_llm_content(response: &OpenAiResponse) -> Result<String, Box<dyn Error + Send + Sync>> {
    if let MessageContent::String { content } = &response.choices[0].message.content {
        Ok(content.clone())
    } else {
        Err(
            Box::new(LLMError::Generic("Invalid content type".to_string()))
                as Box<dyn Error + Send + Sync>,
        )
    }
}

/// Extract fenced content from a string
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

/// Try to extract fenced content from an OpenAI response
/// Returns message content if extraction succeeds, error otherwise
fn try_extract_from_response(
    response: &OpenAiResponse,
    fence_type: Option<&str>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    if response.choices.is_empty() {
        println!("Response contains no choices");
        return Err(Box::new(LLMError::Generic(
            "No choices found in response".to_string(),
        )));
    }

    if response.choices[0].finish_reason == "length" {
        println!("Response was truncated (finish_reason: length)");
        return Err(Box::new(LLMError::Generic(
            "Response was truncated (finish_reason: length)".to_string(),
        )));
    }

    let content = get_llm_content(response)?;
    if content.trim().is_empty() {
        println!("Content is empty");
        return Err(Box::new(LLMError::Generic("Content is empty".to_string())));
    }
    let extracted = extract_fenced_content(&content, fence_type).ok_or_else(|| {
        println!("No content could be extracted from response content");
        Box::new(LLMError::Generic(
            "No content could be extracted from response content".to_string(),
        ))
    })?;
    Ok(extracted)
}

pub async fn try_extract_from_llm(
    messages: Vec<Message>,
    fence_type: Option<&str>,
    fallback_content: Option<String>,
    llm_processing: LlmProcessing,
    tracer: &opentelemetry::global::BoxedTracer,
    parent_context: &Context,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut span = tracer.start_with_context("try_extract_from_llm", parent_context);

    if let Some(segment_id) = parent_context.baggage().get("segment_id") {
        span.set_attribute(opentelemetry::KeyValue::new(
            "segment_id",
            segment_id.to_string(),
        ));
    }
    if let Some(segment_type) = parent_context.baggage().get("segment_type") {
        span.set_attribute(opentelemetry::KeyValue::new(
            "segment_type",
            segment_type.to_string(),
        ));
    }

    let ctx = parent_context.with_span(span);

    let llm_config = LlmConfig::from_env().unwrap();
    let model = llm_config.get_model(llm_processing.model_id)?;
    let fallback_model = llm_config.get_fallback_model(llm_processing.fallback_strategy)?;

    match process_openai_request(
        model,
        fallback_model.clone(),
        messages.clone(),
        llm_processing.max_completion_tokens,
        Some(llm_processing.temperature),
        None,
        fence_type,
        tracer,
        &ctx,
    )
    .await
    {
        Ok(response) => Ok(response),
        Err(e) => {
            if let Some(fallback_content) = fallback_content {
                ctx.span()
                    .set_attribute(opentelemetry::KeyValue::new("using_fallback_content", true));
                return Ok(fallback_content);
            }
            ctx.span()
                .set_status(opentelemetry::trace::Status::error(e.to_string()));
            ctx.span().record_error(e.as_ref());
            ctx.span()
                .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            Err(e)
        }
    }
}
