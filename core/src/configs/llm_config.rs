use crate::models::chunkr::open_ai::{ Message, MessageContent };
use base64::{engine::general_purpose, Engine as _};
use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub fallback_model: Option<String>,
    #[serde(default = "default_key")]
    pub key: String,
    #[serde(default = "default_model")]
    pub model: String,
    pub ocr_key: Option<String>,
    pub ocr_model: Option<String>,
    pub ocr_url: Option<String>,
    pub structured_extraction_key: Option<String>,
    pub structured_extraction_model: Option<String>,
    pub structured_extraction_url: Option<String>,
    #[serde(default = "default_url")]
    pub url: String,
}

fn default_key() -> String {
    "".to_string()
}

fn default_model() -> String {
    "gpt-4o".to_string()
}

fn default_url() -> String {
    "https://api.openai.com/v1/chat/completions".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();
        ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("LLM").separator("__"))
            .build()?
            .try_deserialize::<Self>()
    }
}

macro_rules! prompt_templates {
    ($($name:expr),* $(,)?) => {
        &[
            $(
                ($name, include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/utils/prompts/", $name, ".txt")))
            ),*
        ]
    };
}

const PROMPT_TEMPLATES: &[(&str, &str)] = prompt_templates![
    "formula",
    "html_caption",
    "html_footnote",
    "html_list_item",
    "html_page_footer",
    "html_page_header",
    "html_page",
    "html_picture",
    "html_section_header",
    "html_table",
    "html_text",
    "html_title",
    "llm_segment",
    "md_caption",
    "md_footnote",
    "md_list_item",
    "md_page_footer",
    "md_page_header",
    "md_page",
    "md_picture",
    "md_section_header",
    "md_table",
    "md_text",
    "md_title",
    "structured_extraction_system",
    "structured_extraction_user"
];

fn get_template(prompt_name: &str) -> Result<String, std::io::Error> {
    PROMPT_TEMPLATES
        .iter()
        .find(|&&(name, _)| name == prompt_name)
        .map(|(_, content)| content.to_string())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Prompt '{}' not found", prompt_name),
            )
        })
}

fn fill_prompt(template: &str, values: &std::collections::HashMap<String, String>) -> String {
    let mut result = template.to_string();

    result = result.replace("\\{", r"\u005c\u007b");
    result = result.replace("\\}", r"\u005c\u007d");

    for (key, value) in values {
        result = result.replace(&format!("{{{}}}", key), value);
    }
    result = result.replace(r"\u005c\u007b", "{");
    result = result.replace(r"\u005c\u007d", "}");

    result
}

pub fn get_prompt(
    prompt_name: &str,
    values: &HashMap<String, String>,
) -> Result<String, std::io::Error> {
    let template = get_template(prompt_name)?;
    let filled_prompt = fill_prompt(&template, values);
    Ok(filled_prompt)
}

macro_rules! json_prompt_templates {
    ($($name:expr),* $(,)?) => {
        &[
            $(
                ($name, include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/utils/prompts/", $name, ".json")))
            ),*
        ]
    };
}

const JSON_PROMPT_TEMPLATES: &[(&str, &str)] = json_prompt_templates![
    "formula_json"
];

fn get_json_template(prompt_name: &str) -> Result<Value, std::io::Error> {
    JSON_PROMPT_TEMPLATES
        .iter()
        .find(|&&(name, _)| name == prompt_name)
        .map(|(_, content)| {
            serde_json::from_str(content).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to parse JSON template: {}", e),
                )
            })
        })
        .unwrap_or_else(|| {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("JSON prompt template '{}' not found", prompt_name),
            ))
        })
}

static TEMPLATE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\{\{([^{}]+)\}\}").unwrap());

fn process_template_value(value: &Value, values: &HashMap<String, String>) -> Result<Value, std::io::Error> {
    match value {
        Value::String(s) => {
            let processed = TEMPLATE_REGEX.replace_all(s, |caps: &regex::Captures| {
                let key = caps.get(1).unwrap().as_str().trim();
                values.get(key).cloned().unwrap_or_else(|| format!("{{{{{}}}}}", key))
            }).to_string();
            
            if processed.starts_with("file://") {
                let path = &processed[7..];
                let file_path = Path::new(path);
                if file_path.exists() {
                    let mut file = File::open(file_path)?;
                    let mut buffer = Vec::new();
                    file.read_to_end(&mut buffer)?;
                    
                    let mime_type = match file_path.extension().and_then(|ext| ext.to_str()) {
                        Some("jpg") | Some("jpeg") => "image/jpeg",
                        Some("png") => "image/png",
                        _ => "image/jpeg", 
                    };
                    
                    let base64_image = general_purpose::STANDARD.encode(&buffer);
                    Ok(Value::String(format!("data:{};base64,{}", mime_type, base64_image)))
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("File not found: {}", path),
                    ))
                }
            } else {
                Ok(Value::String(processed))
            }
        },
        Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                new_map.insert(k.clone(), process_template_value(v, values)?);
            }
            Ok(Value::Object(new_map))
        },
        Value::Array(arr) => {
            let mut new_arr = Vec::new();
            for item in arr {
                new_arr.push(process_template_value(item, values)?);
            }
            Ok(Value::Array(new_arr))
        },
        _ => Ok(value.clone()),
    }
}

fn fill_json_template(
    template: &Value,
    values: &HashMap<String, String>,
) -> Result<Vec<Message>, std::io::Error> {
    if let Value::Object(obj) = template {
        if let Some(Value::Array(messages_arr)) = obj.get("messages") {
            let mut messages = Vec::new();
            
            for message_value in messages_arr {
                if let Value::Object(message_obj) = message_value {
                    let processed_message = process_template_value(&Value::Object(message_obj.clone()), values)?;
                    
                    let message: Message = serde_json::from_value(processed_message)
                        .map_err(|e| std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Failed to deserialize message: {}", e)
                        ))?;
                    
                    messages.push(message);
                }
            }
            
            return Ok(messages);
        }
    }
    
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "Invalid template format: expected 'messages' array".to_string(),
    ))
}


// TODO: Remove backward compatibility as it cant be converted to messages properly
pub fn get_prompt_messages(
    prompt_name: &str,
    values: &HashMap<String, String>,
) -> Result<Vec<Message>, std::io::Error> {
    match get_json_template(prompt_name) {
        Ok(template) => fill_json_template(&template, values),
        Err(_) => {
            // Fall back to text template for backward compatibility
            get_template(prompt_name).map(|template| {
                let filled_prompt = fill_prompt(&template, values);
                vec![Message {
                    role: "user".to_string(),
                    content: MessageContent::String { content: filled_prompt },
                }]
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    #[tokio::test]
    async fn test_get_template() -> Result<(), std::io::Error> {
        let prompt = get_template("structured_extraction").unwrap();
        println!("Prompt: {}", prompt);
        Ok(())
    }

    #[tokio::test]
    async fn test_fill_prompt_with_values() -> Result<(), std::io::Error> {
        let mut values = HashMap::new();
        values.insert("name".to_string(), "Invoice Number".to_string());
        values.insert(
            "description".to_string(),
            "The unique identifier for this invoice".to_string(),
        );
        values.insert("field_type".to_string(), "string".to_string());
        values.insert("context".to_string(), "Invoice #12345...".to_string());
        let filled_prompt = get_prompt("structured_extraction", &values)?;
        println!("{}", filled_prompt);
        Ok(())
    }

    #[tokio::test]
    async fn test_fill_prompt_with_values_table() -> Result<(), std::io::Error> {
        let filled_prompt = get_prompt("table", &HashMap::new())?;
        println!("{}", filled_prompt);
        Ok(())
    }
}
