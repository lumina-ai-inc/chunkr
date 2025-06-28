use crate::models::genai;
use serde::{Deserialize, Serialize};
use std::error::Error;
use uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Choice {
    pub index: i32,
    pub message: Message,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    #[serde(flatten)]
    pub content: MessageContent,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    String { content: String },
    Array { content: Vec<ContentPart> },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Usage {
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
    pub total_tokens: Option<i32>,
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<i32>,
    pub accepted_prediction_tokens: Option<i32>,
    pub rejected_prediction_tokens: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAiResponse {
    pub choices: Vec<Choice>,
    pub created: i64,
    #[serde(default = "generate_uuid")]
    pub id: String,
    pub model: String,
    pub object: String,
    pub system_fingerprint: Option<String>,
    pub usage: Usage,
}

fn generate_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchema },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JsonSchema {
    pub name: String,
    pub description: Option<String>,
    pub schema: serde_json::Value,
    pub strict: Option<bool>,
}

/// Converts an OpenAI request to a Gemini request
///
/// The json_schema must be added separately to the generation_config.response_schema field.
impl From<OpenAiRequest> for genai::GenerateContentRequest {
    fn from(request: OpenAiRequest) -> Self {
        let contents = request.messages.into_iter().map(|msg| msg.into()).collect();

        let mut generation_config = genai::GenerationConfig::default();

        if let Some(max_tokens) = request.max_completion_tokens {
            generation_config.max_output_tokens = Some(max_tokens as i32);
        }

        if let Some(temp) = request.temperature {
            generation_config.temperature = Some(temp as f64);
        }

        // Handle structured output
        if let Some(response_format) = request.response_format {
            match response_format {
                ResponseFormat::JsonObject => {
                    generation_config.response_mime_type = Some("application/json".to_string());
                }
                ResponseFormat::JsonSchema { .. } => {
                    generation_config.response_mime_type = Some("application/json".to_string());
                }
                ResponseFormat::Text => {
                    // Default text format
                }
            }
        }

        Self {
            contents,
            tools: None,
            tool_config: None,
            safety_settings: None,
            system_instruction: None,
            generation_config: Some(generation_config),
            cached_content: None,
        }
    }
}

impl From<Message> for genai::Content {
    fn from(message: Message) -> Self {
        let role = Some(message.role);
        let parts = match message.content {
            MessageContent::String { content } => {
                vec![genai::Part::Text { text: content }]
            }
            MessageContent::Array { content } => {
                content.into_iter().map(|part| part.into()).collect()
            }
        };

        Self { role, parts }
    }
}

impl From<ContentPart> for genai::Part {
    fn from(part: ContentPart) -> Self {
        match part.content_type.as_str() {
            "text" => genai::Part::Text {
                text: part.text.unwrap_or_default(),
            },
            "image_url" => {
                if let Some(image_url) = part.image_url {
                    // Check if this is a data URI (base64 encoded image)
                    if image_url.url.starts_with("data:") {
                        // Parse data URI: data:[<mediatype>][;base64],<data>
                        if let Some((header, data)) = image_url.url.split_once(',') {
                            // Extract MIME type from header
                            let mime_type = if let Some(media_part) = header.strip_prefix("data:") {
                                if let Some((media_type, _encoding)) = media_part.split_once(';') {
                                    media_type.to_string()
                                } else {
                                    media_part.to_string()
                                }
                            } else {
                                "image/jpeg".to_string() // fallback
                            };

                            genai::Part::InlineData {
                                inline_data: genai::Blob {
                                    mime_type,
                                    data: data.to_string(),
                                },
                            }
                        } else {
                            // Malformed data URI, fallback to text
                            genai::Part::Text {
                                text: "Invalid image data URI".to_string(),
                            }
                        }
                    } else {
                        // Regular file URI
                        genai::Part::FileData {
                            file_data: genai::FileData {
                                mime_type: "image/jpeg".to_string(), // Default, could be improved
                                file_uri: image_url.url,
                            },
                        }
                    }
                } else {
                    genai::Part::Text {
                        text: "".to_string(),
                    }
                }
            }
            _ => genai::Part::Text {
                text: part.text.unwrap_or_default(),
            },
        }
    }
}

impl TryFrom<genai::GenerateContentResponse> for OpenAiResponse {
    type Error = Box<dyn Error + Send + Sync>;

    fn try_from(response: genai::GenerateContentResponse) -> Result<Self, Self::Error> {
        let choices = response
            .candidates
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(
                |(index, candidate)| -> Result<Choice, Box<dyn Error + Send + Sync>> {
                    let message = candidate.content.ok_or("No content in candidate")?.into();
                    let finish_reason = candidate
                        .finish_reason
                        .map(|reason| format!("{reason:?}").to_lowercase())
                        .unwrap_or_else(|| "stop".to_string());

                    Ok(Choice {
                        index: index as i32,
                        message,
                        logprobs: None,
                        finish_reason,
                    })
                },
            )
            .collect::<Result<Vec<_>, _>>()?;

        let usage = Usage {
            prompt_tokens: response
                .usage_metadata
                .as_ref()
                .and_then(|u| u.prompt_token_count),
            completion_tokens: response
                .usage_metadata
                .as_ref()
                .and_then(|u| u.candidates_token_count),
            total_tokens: response
                .usage_metadata
                .as_ref()
                .and_then(|u| u.total_token_count),
            completion_tokens_details: None,
        };

        Ok(Self {
            choices,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            id: response.response_id.unwrap_or_else(generate_uuid),
            model: response
                .model_version
                .unwrap_or_else(|| "gemini".to_string()),
            object: "chat.completion".to_string(),
            system_fingerprint: None,
            usage,
        })
    }
}

impl From<genai::Content> for Message {
    fn from(content: genai::Content) -> Self {
        let role = content.role.unwrap_or_else(|| "assistant".to_string());
        let message_content = if content.parts.len() == 1 {
            if let genai::Part::Text { text } = &content.parts[0] {
                MessageContent::String {
                    content: text.clone(),
                }
            } else {
                MessageContent::Array {
                    content: content.parts.into_iter().map(|part| part.into()).collect(),
                }
            }
        } else {
            MessageContent::Array {
                content: content.parts.into_iter().map(|part| part.into()).collect(),
            }
        };

        Self {
            role,
            content: message_content,
        }
    }
}

impl From<genai::Part> for ContentPart {
    fn from(part: genai::Part) -> Self {
        match part {
            genai::Part::Text { text } => ContentPart {
                content_type: "text".to_string(),
                text: Some(text),
                image_url: None,
            },
            genai::Part::FileData { file_data } => ContentPart {
                content_type: "image_url".to_string(),
                text: None,
                image_url: Some(ImageUrl {
                    url: file_data.file_uri,
                }),
            },
            genai::Part::InlineData { inline_data } => ContentPart {
                content_type: "image_url".to_string(),
                text: None,
                image_url: Some(ImageUrl {
                    url: format!("data:{};base64,{}", inline_data.mime_type, inline_data.data),
                }),
            },
            _ => ContentPart {
                content_type: "text".to_string(),
                text: Some("".to_string()),
                image_url: None,
            },
        }
    }
}

/// Converts a schemars-generated JSON schema to OpenAI-compatible format
///
/// This function removes unsupported fields and formats that OpenAI's structured output doesn't support:
/// - Removes `$schema` and `title` fields
/// - Removes `format` fields (like "uint32", "float") from properties
/// - Adds `additionalProperties: false` recursively to all objects for strict mode
/// - Preserves descriptions from doc comments
pub fn convert_schema_to_openai_format(mut schema_json: serde_json::Value) -> serde_json::Value {
    convert_schema_to_openai_format_recursive(&mut schema_json);
    schema_json
}

/// Recursively processes a JSON schema to make it OpenAI-compatible
fn convert_schema_to_openai_format_recursive(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(obj) => {
            // Remove unsupported fields at root level
            obj.remove("$schema");
            obj.remove("title");

            // Remove unsupported format fields
            obj.remove("format");

            // Check if this is an object type schema
            if let Some(schema_type) = obj.get("type").and_then(|t| t.as_str()) {
                if schema_type == "object" {
                    // Add additionalProperties: false for strict mode
                    obj.insert(
                        "additionalProperties".to_string(),
                        serde_json::Value::Bool(false),
                    );
                }
            } else if obj.contains_key("properties") {
                // If no explicit type but has properties, assume it's an object
                obj.insert(
                    "additionalProperties".to_string(),
                    serde_json::Value::Bool(false),
                );
            }

            // Recursively process all nested values
            for (_key, nested_value) in obj.iter_mut() {
                convert_schema_to_openai_format_recursive(nested_value);
            }
        }
        serde_json::Value::Array(arr) => {
            // Recursively process array elements
            for item in arr.iter_mut() {
                convert_schema_to_openai_format_recursive(item);
            }
        }
        _ => {
            // For primitive types, no processing needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_convert_schema_to_openai_format_with_nested_objects() {
        // Test schema with nested objects to ensure additionalProperties is added recursively
        let schema = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test Schema",
            "type": "object",
            "properties": {
                "simple_field": {
                    "type": "string",
                    "format": "some_format"
                },
                "nested_object": {
                    "type": "object",
                    "properties": {
                        "inner_field": {
                            "type": "string"
                        },
                        "deeply_nested": {
                            "type": "object",
                            "properties": {
                                "deep_field": {
                                    "type": "number",
                                    "format": "float"
                                }
                            }
                        }
                    }
                },
                "array_field": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "array_item_field": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
        });

        let converted = convert_schema_to_openai_format(schema);

        // Root object should have additionalProperties: false
        assert_eq!(converted.get("additionalProperties"), Some(&json!(false)));

        // Should remove unsupported fields from root
        assert!(converted.get("$schema").is_none());
        assert!(converted.get("title").is_none());

        // Nested object should have additionalProperties: false
        let nested_obj = converted
            .get("properties")
            .and_then(|p| p.get("nested_object"));
        assert_eq!(
            nested_obj.and_then(|o| o.get("additionalProperties")),
            Some(&json!(false))
        );

        // Deeply nested object should have additionalProperties: false
        let deeply_nested = nested_obj
            .and_then(|o| o.get("properties"))
            .and_then(|p| p.get("deeply_nested"));
        assert_eq!(
            deeply_nested.and_then(|o| o.get("additionalProperties")),
            Some(&json!(false))
        );

        // Array items object should have additionalProperties: false
        let array_items = converted
            .get("properties")
            .and_then(|p| p.get("array_field"))
            .and_then(|a| a.get("items"));
        assert_eq!(
            array_items.and_then(|o| o.get("additionalProperties")),
            Some(&json!(false))
        );

        // Format fields should be removed
        let simple_field = converted
            .get("properties")
            .and_then(|p| p.get("simple_field"));
        assert!(simple_field.and_then(|f| f.get("format")).is_none());

        let deep_field = deeply_nested
            .and_then(|o| o.get("properties"))
            .and_then(|p| p.get("deep_field"));
        assert!(deep_field.and_then(|f| f.get("format")).is_none());
    }

    #[test]
    fn test_data_uri_parsing() {
        // Test data URI with PNG image
        let data_uri_part = ContentPart {
            content_type: "image_url".to_string(),
            text: None,
            image_url: Some(ImageUrl {
                url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77yQAAAABJRU5ErkJggg==".to_string(),
            }),
        };

        let genai_part: genai::Part = data_uri_part.into();

        // Should be InlineData, not FileData
        match genai_part {
            genai::Part::InlineData { inline_data } => {
                assert_eq!(inline_data.mime_type, "image/png");
                assert_eq!(inline_data.data, "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77yQAAAABJRU5ErkJggg==");
            }
            _ => panic!("Expected InlineData for data URI"),
        }
    }

    #[test]
    fn test_regular_uri_parsing() {
        // Test regular file URI
        let file_uri_part = ContentPart {
            content_type: "image_url".to_string(),
            text: None,
            image_url: Some(ImageUrl {
                url: "https://example.com/image.jpg".to_string(),
            }),
        };

        let genai_part: genai::Part = file_uri_part.into();

        // Should be FileData
        match genai_part {
            genai::Part::FileData { file_data } => {
                assert_eq!(file_data.file_uri, "https://example.com/image.jpg");
                assert_eq!(file_data.mime_type, "image/jpeg"); // Default fallback
            }
            _ => panic!("Expected FileData for regular URI"),
        }
    }

    #[test]
    fn test_data_uri_with_different_mime_type() {
        // Test data URI with JPEG image
        let data_uri_part = ContentPart {
            content_type: "image_url".to_string(),
            text: None,
            image_url: Some(ImageUrl {
                url: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD//gA7Q1JFQVR".to_string(),
            }),
        };

        let genai_part: genai::Part = data_uri_part.into();

        match genai_part {
            genai::Part::InlineData { inline_data } => {
                assert_eq!(inline_data.mime_type, "image/jpeg");
                assert_eq!(inline_data.data, "/9j/4AAQSkZJRgABAQEAYABgAAD//gA7Q1JFQVR");
            }
            _ => panic!("Expected InlineData for JPEG data URI"),
        }
    }
}
