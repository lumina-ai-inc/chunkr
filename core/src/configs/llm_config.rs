use crate::models::chunkr::open_ai::Message;
use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
                ($name, include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/utils/prompts/", $name, ".json")))
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
];

fn load_prompt_template(prompt_name: &str) -> Result<String, std::io::Error> {
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

fn substitute_template_placeholders(
    template_json: &str,
    values: &HashMap<String, String>,
) -> Result<String, serde_json::Error> {
    let mut template = template_json.to_string();

    // Replace all placeholder values in the JSON string
    for (key, value) in values {
        // Escape any special characters in the value for JSON compatibility
        let escaped_value = serde_json::to_string(value)?;
        // Remove the surrounding quotes that to_string adds
        let escaped_value = &escaped_value[1..escaped_value.len() - 1];

        template = template.replace(&format!("{{{}}}", key), escaped_value);
    }

    Ok(template)
}

pub fn create_messages_from_template(
    template_name: &str,
    values: &HashMap<String, String>,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let template_json = load_prompt_template(template_name)?;
    let filled_json = substitute_template_placeholders(&template_json, values)?;
    let messages: Vec<Message> = serde_json::from_str(&filled_json)?;
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test template JSON that we can use directly in tests
    const TEST_TEMPLATE_JSON: &str = r#"[
        {
            "role": "system",
            "content": "You are a helpful AI assistant for {purpose}."
        },
        {
            "role": "user",
            "content": "Please process this {content_type} data: {data}"
        }
    ]"#;

    // Helper function to create a test template without file dependencies
    fn get_test_template() -> String {
        TEST_TEMPLATE_JSON.to_string()
    }

    #[tokio::test]
    async fn test_load_template() -> Result<(), Box<dyn std::error::Error>> {
        let prompt = load_prompt_template("formula")?;
        println!("Template JSON: {}", prompt);
        Ok(())
    }

    #[tokio::test]
    async fn test_substitute_template_placeholders() -> Result<(), Box<dyn std::error::Error>> {
        let mut values = HashMap::new();
        values.insert("purpose".to_string(), "data extraction".to_string());
        values.insert("content_type".to_string(), "table".to_string());
        values.insert("data".to_string(), "Row 1: 42, Row 2: 73".to_string());

        let template = get_test_template();
        let filled_json = substitute_template_placeholders(&template, &values)?;
        println!("Filled template: {}", filled_json);

        // Parse to verify it's valid JSON
        let parsed: Vec<Message> = serde_json::from_str(&filled_json)?;
        assert_eq!(parsed.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_create_messages_from_template() -> Result<(), Box<dyn std::error::Error>> {
        let mut values = HashMap::new();
        values.insert(
            "image_url".to_string(),
            "https://example.com/image.jpg".to_string(),
        );
        let messages = create_messages_from_template("md_table", &values)?;
        println!("Message: {:?}", messages);
        Ok(())
    }
}
