use crate::configs::llm_config::create_messages_from_template;
use crate::configs::llm_config::{Config as LlmConfig, LlmModel};
use crate::configs::otel_config;
use crate::models::genai::{GenerateContentRequest, GenerateContentResponse};
use crate::models::llm::LlmProcessing;
use crate::models::llm::{JsonSchemaDefinition, LlmProvider};
use crate::models::open_ai::{
    Message, MessageContent, OpenAiRequest, OpenAiResponse, ResponseFormat,
};
use crate::utils::rate_limit::{get_llm_rate_limiter, LLM_TIMEOUT, TOKEN_TIMEOUT};
use crate::utils::retry::retry_with_backoff_conditional;
use futures::TryFutureExt;
use opentelemetry::baggage::BaggageExt;
use opentelemetry::trace::{Span, TraceContextExt, Tracer};
use opentelemetry::Context;
use schemars::JsonSchema as SchemarsJsonSchema;
use std::error::Error;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("Error parsing JSON: {error}")]
    JsonParseError { error: String, response: String },

    #[error("{0}")]
    Generic(String),

    #[error("Non-retryable error: {0}")]
    NonRetryable(String),
}

/// Check if an error should be retried
#[allow(clippy::borrowed_box)]
fn is_retryable_error(error: &Box<dyn Error + Send + Sync>) -> bool {
    // Check if the error is a NonRetryable LLM error
    if let Some(llm_error) = error.downcast_ref::<LLMError>() {
        !matches!(llm_error, LLMError::NonRetryable(_))
    } else {
        // For non-LLM errors, default to retryable
        true
    }
}

pub async fn open_ai_call(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    schema_definition: Option<JsonSchemaDefinition>,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    println!("OpenAI call with model: {model:?}");

    // Convert schema definition to OpenAI format if provided
    let response_format = if let Some(schema_def) = schema_definition {
        Some(ResponseFormat::JsonSchema {
            json_schema: crate::models::open_ai::JsonSchema {
                name: schema_def.name,
                description: schema_def.description,
                schema: crate::models::open_ai::convert_schema_to_openai_format(schema_def.schema),
                strict: Some(true),
            },
        })
    } else {
        None
    };

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
        .header("Authorization", format!("Bearer {key}"));

    if let Some(Some(timeout_value)) = LLM_TIMEOUT.get() {
        openai_request = openai_request.timeout(std::time::Duration::from_secs(*timeout_value));
    }

    let response = openai_request.json(&request).send().await?;

    let response = if response.status().is_success() {
        response
    } else {
        let status = response.status();
        let error_text = response.text().await?;
        let error_message = format!("HTTP {status}: {error_text}");

        // 400-level errors (client errors) should not be retried
        if status.as_u16() >= 400 && status.as_u16() < 500 {
            return Err(Box::new(LLMError::NonRetryable(error_message)));
        } else {
            return Err(Box::new(LLMError::Generic(error_message)));
        }
    };

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

pub async fn genai_call(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    schema_definition: Option<JsonSchemaDefinition>,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    println!("Gemini call with model: {model:?}");

    // Create a basic OpenAI request for conversion
    let openai_request = OpenAiRequest {
        model: model.clone(),
        messages,
        max_completion_tokens,
        temperature,
        response_format: None,
    };

    let mut genai_request: GenerateContentRequest = openai_request.into();

    // Add schema to generation config if provided
    if let Some(schema_def) = schema_definition {
        if let Some(ref mut gen_config) = genai_request.generation_config {
            gen_config.response_mime_type = Some("application/json".to_string());
            gen_config.response_schema = Some(
                crate::models::genai::convert_schema_to_genai_format(schema_def.schema)?,
            );
        }
    }

    let client = reqwest::Client::new();
    let gemini_url = format!("{url}/models/{model}:generateContent?key={key}");
    let mut gemini_request = client
        .post(gemini_url)
        .header("Content-Type", "application/json");

    if let Some(Some(timeout_value)) = LLM_TIMEOUT.get() {
        gemini_request = gemini_request.timeout(std::time::Duration::from_secs(*timeout_value));
    }

    let response = gemini_request.json(&genai_request).send().await?;

    let response = if response.status().is_success() {
        response
    } else {
        let status = response.status();
        let error_text = response.text().await?;
        let error_message = format!("HTTP {status}: {error_text}");

        // 400-level errors (client errors) should not be retried
        if status.as_u16() >= 400 && status.as_u16() < 500 {
            return Err(Box::new(LLMError::NonRetryable(error_message)));
        } else {
            return Err(Box::new(LLMError::Generic(error_message)));
        }
    };

    let text = response.text().await?;
    let genai_response: GenerateContentResponse = match serde_json::from_str(&text) {
        Ok(parsed) => parsed,
        Err(e) => {
            return Err(Box::new(LLMError::JsonParseError {
                error: e.to_string(),
                response: text.trim().to_string(),
            }));
        }
    };

    // Convert Gemini response back to OpenAI format
    genai_response.try_into()
}

#[allow(clippy::too_many_arguments)]
pub async fn llm_call_router(
    url: String,
    key: String,
    model: String,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    schema_definition: Option<JsonSchemaDefinition>,
    tracer: &opentelemetry::global::BoxedTracer,
    parent_context: &Context,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    let provider = LlmProvider::from_url(&url);
    let span = tracer.start_with_context(
        otel_config::SpanName::LlmCallRouter.to_string(),
        parent_context,
    );
    let ctx = parent_context.with_span(span);
    ctx.span()
        .set_attribute(opentelemetry::KeyValue::new("model", model.clone()));
    ctx.span()
        .set_attribute(opentelemetry::KeyValue::new("provider_url", url.clone()));
    ctx.span().set_attribute(opentelemetry::KeyValue::new(
        "provider",
        provider.to_string(),
    ));

    match provider {
        LlmProvider::Genai => {
            genai_call(
                url,
                key,
                model,
                messages,
                max_completion_tokens,
                temperature,
                schema_definition,
            )
            .await
        }
        LlmProvider::OpenAI => {
            open_ai_call(
                url,
                key,
                model,
                messages,
                max_completion_tokens,
                temperature,
                schema_definition,
            )
            .await
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

/// Process an OpenAI request with rate limiting and retrying on failure.
async fn open_ai_call_handler(
    model: LlmModel,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    schema_definition: Option<JsonSchemaDefinition>,
    tracer: &opentelemetry::global::BoxedTracer,
    parent_context: &Context,
) -> Result<OpenAiResponse, Box<dyn Error + Send + Sync>> {
    let rate_limiter = get_llm_rate_limiter(&model.id)?;
    let mut span = tracer.start_with_context(
        otel_config::SpanName::OpenAiCall.to_string(),
        parent_context,
    );
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

    match llm_call_router(
        model.provider_url.clone(),
        model.api_key.clone(),
        model.model.clone(),
        messages.clone(),
        max_completion_tokens,
        temperature,
        schema_definition,
        tracer,
        &ctx,
    )
    .await
    {
        Ok(response) => Ok(response),
        Err(e) => {
            if let Some(LLMError::JsonParseError { response, .. }) = e.downcast_ref::<LLMError>() {
                let attributes = otel_config::extract_llm_error_attributes(response);
                for attr in attributes {
                    ctx.span().set_attribute(attr);
                }
            }
            Err(e)
        }
    }
    .inspect_err(|e| {
        println!("Error: {e}");
        ctx.span()
            .set_status(opentelemetry::trace::Status::error(e.to_string()));
        ctx.span().record_error(e.as_ref());
        ctx.span()
            .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
    })
}

#[allow(clippy::too_many_arguments)]
async fn try_extract_from_open_ai_response(
    model: LlmModel,
    messages: Vec<Message>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    schema_definition: Option<JsonSchemaDefinition>,
    fence_type: Option<&str>,
    tracer: &opentelemetry::global::BoxedTracer,
    parent_context: &Context,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut span = tracer.start_with_context(
        otel_config::SpanName::TryExtractFromOpenAiResponse.to_string(),
        parent_context,
    );
    span.set_attribute(opentelemetry::KeyValue::new("model", model.model.clone()));
    span.set_attribute(opentelemetry::KeyValue::new(
        "provider_url",
        model.provider_url.clone(),
    ));
    if let Some(temperature) = temperature {
        span.set_attribute(opentelemetry::KeyValue::new(
            "temperature",
            temperature as f64,
        ));
    }
    if let Some(max_completion_tokens) = max_completion_tokens {
        span.set_attribute(opentelemetry::KeyValue::new(
            "max_completion_tokens",
            max_completion_tokens as f64,
        ));
    }
    if let Some(_schema_def) = schema_definition.as_ref() {
        span.set_attribute(opentelemetry::KeyValue::new("has_schema", true));
    }

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

    open_ai_call_handler(
        model,
        messages,
        max_completion_tokens,
        temperature,
        schema_definition,
        tracer,
        parent_context,
    )
    .await
    .and_then(|response| try_extract_from_response(&response, fence_type))
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
    match fence_type {
        None => {
            // No fencing, return content as-is
            Some(content.trim().to_string())
        }
        Some(ft) => {
            let split_pattern = if ft.is_empty() {
                "```".to_string()
            } else {
                format!("```{ft}")
            };

            content
                .split(&split_pattern)
                .nth(1)
                .and_then(|content| content.split("```").next())
                .map(|content| content.trim().to_string())
        }
    }
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

pub async fn llm_handler(
    llm_processing: LlmProcessing,
    fallback_content: Option<String>,
    messages: Vec<Message>,
    schema_definition: Option<JsonSchemaDefinition>,
    fence_type: Option<&str>,
    tracer: &opentelemetry::global::BoxedTracer,
    ctx: &Context,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let model_id = llm_processing.model_id.clone().ok_or_else(|| {
        Box::new(LLMError::Generic("Model ID is required".to_string()))
            as Box<dyn Error + Send + Sync>
    })?;
    let fallback_strategy = llm_processing.fallback_strategy;
    let max_completion_tokens = llm_processing.max_completion_tokens;
    let temperature = llm_processing.temperature;

    retry_with_backoff_conditional(
        || async {
            let messages_clone = messages.clone();
            let llm_config = LlmConfig::from_env().unwrap();
            let model = llm_config.get_model(Some(model_id.clone()))?;
            let fallback_model = llm_config.get_fallback_model(fallback_strategy.clone())?;
            let ctx_clone = ctx.clone();
            let schema_definition_clone = schema_definition.clone();
            println!("Fallback content exists: {:?}", fallback_content.is_some());

            try_extract_from_open_ai_response(
                model,
                messages_clone.clone(),
                max_completion_tokens,
                Some(temperature),
                schema_definition_clone.clone(),
                fence_type,
                tracer,
                ctx,
            )
            .or_else(|e| async move {
                if is_retryable_error(&e) && fallback_model.is_some() {
                    let fallback_model = fallback_model.unwrap();
                    try_extract_from_open_ai_response(
                        fallback_model,
                        messages_clone,
                        max_completion_tokens,
                        Some(temperature),
                        schema_definition_clone.clone(),
                        fence_type,
                        tracer,
                        &ctx_clone,
                    )
                    .await
                } else {
                    Err(e)
                }
            })
            .await
            .or_else(|e| {
                if let Some(fallback_content) = fallback_content.clone() {
                    ctx.span().set_attribute(opentelemetry::KeyValue::new(
                        "using_fallback_content",
                        true,
                    ));
                    Ok(fallback_content)
                } else {
                    Err(e)
                }
            })
            .inspect_err(|e| {
                ctx.span()
                    .set_status(opentelemetry::trace::Status::error(e.to_string()));
                ctx.span().record_error(e.as_ref());
                ctx.span()
                    .set_attribute(opentelemetry::KeyValue::new("error", e.to_string()));
            })
        },
        is_retryable_error,
    )
    .await
}

pub async fn try_extract_from_llm(
    messages: Vec<Message>,
    fence_type: Option<&str>,
    fallback_content: Option<String>,
    llm_processing: LlmProcessing,
    schema_definition: Option<JsonSchemaDefinition>,
    tracer: &opentelemetry::global::BoxedTracer,
    parent_context: &Context,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let mut span = tracer.start_with_context(
        otel_config::SpanName::TryExtractFromLlm.to_string(),
        parent_context,
    );
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
    llm_handler(
        llm_processing,
        fallback_content,
        messages,
        schema_definition,
        fence_type,
        tracer,
        &ctx,
    )
    .await
}

/// Structured extraction helper that handles schema generation and parsing
///
/// This is a convenience function that:
/// 1. Creates messages from a template
/// 2. Generates the OpenAI schema for the target struct
/// 3. Makes the LLM call with structured output
/// 4. Parses and returns the result as the target struct
///
/// # Example
/// ```rust
/// let result: MyStruct = extract_structured_from_template(
///     "my_template",
///     &values,
///     llm_processing,
///     &tracer,
///     &context,
/// ).await?;
/// ```
pub async fn structured_output_from_template<T>(
    name: &str,
    description: Option<String>,
    template_name: &str,
    values: &std::collections::HashMap<String, String>,
    llm_processing: LlmProcessing,
    tracer: &opentelemetry::global::BoxedTracer,
    parent_context: &Context,
) -> Result<T, Box<dyn Error + Send + Sync>>
where
    T: SchemarsJsonSchema + serde::de::DeserializeOwned,
{
    let messages = create_messages_from_template(template_name, values)?;
    let schema_definition = JsonSchemaDefinition::from_struct::<T>(name.to_string(), description);

    let response = try_extract_from_llm(
        messages,
        None, // No fence type needed for structured JSON
        None, // No fallback content
        llm_processing,
        Some(schema_definition), // Schema definition
        tracer,
        parent_context,
    )
    .await?;

    let parsed: T =
        serde_json::from_str(&response).map_err(|e| -> Box<dyn Error + Send + Sync> {
            Box::new(LLMError::JsonParseError {
                error: e.to_string(),
                response: response.clone(),
            })
        })?;

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::llm::FallbackStrategy;
    use crate::models::open_ai::MessageContent;
    use crate::utils::clients;
    use opentelemetry::global;
    use serde::{Deserialize, Serialize};

    // Define a test struct that will be converted to schema and parsed back
    #[derive(Debug, Serialize, Deserialize, SchemarsJsonSchema, PartialEq)]
    struct BookRecommendation {
        /// The title of the book
        title: String,
        /// The author of the book
        author: String,
        /// The genre of the book (e.g., Science Fiction, Fantasy, Mystery)
        genre: String,
        /// The year the book was published
        year_published: u32,
        /// A brief summary of the book's plot
        summary: String,
        /// Rating out of 10.0
        rating: f32,
    }

    #[tokio::test]
    async fn test_text() {
        clients::initialize().await;
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::String {
                content: "What is the weather like in London?".to_string(),
            },
        }];
        let fence_type = None;
        let fallback_content = None;
        let llm_processing = LlmProcessing {
            model_id: Some("gemini-pro-2.5".to_string()),
            fallback_strategy: FallbackStrategy::None,
            max_completion_tokens: None,
            temperature: 0.0,
        };
        let tracer = global::tracer("test");
        let context = opentelemetry::Context::current();
        let response = try_extract_from_llm(
            messages,
            fence_type,
            fallback_content,
            llm_processing,
            None, // No schema for this test
            &tracer,
            &context,
        )
        .await;
        println!("Response: {response:?}");
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_structured_output() {
        clients::initialize().await;

        // Generate raw schema for the struct
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::String {
                content: "Recommend a science fiction book published in the last 10 years. Provide details including title, author, genre, year published, a brief summary, and your rating out of 10.".to_string(),
            },
        }];

        let llm_processing = LlmProcessing {
            model_id: Some("gemini-pro-2.5".to_string()),
            // model_id: Some("gpt-4.1-mini".to_string()),
            fallback_strategy: FallbackStrategy::None,
            max_completion_tokens: None,
            temperature: 0.0,
        };

        let tracer = global::tracer("test");
        let context = opentelemetry::Context::current();

        // Generate schema definition for the struct
        let schema_definition = JsonSchemaDefinition::from_struct::<BookRecommendation>(
            "book_recommendation".to_string(),
            Some("Book recommendation with details".to_string()),
        );

        // Make the LLM call and get the structured response
        let response = try_extract_from_llm(
            messages,
            None, // No fence type needed for structured JSON
            None, // No fallback content
            llm_processing,
            Some(schema_definition), // Schema definition
            &tracer,
            &context,
        )
        .await;

        println!("Raw LLM Response: {response:?}");
        assert!(response.is_ok(), "LLM call should succeed");

        let response_json = response.unwrap();

        let parsed_book: BookRecommendation = match serde_json::from_str(&response_json) {
            Ok(book) => book,
            Err(e) => {
                println!("Error parsing response: {e}");
                panic!("Failed to parse response into BookRecommendation struct: {e}");
            }
        };
        println!("Parsed book: {parsed_book:?}");
    }
}
