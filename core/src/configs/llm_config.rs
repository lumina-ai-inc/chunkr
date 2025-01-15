use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum_macros::Display;

#[derive(Debug, Serialize, Deserialize, Display)]
#[serde(rename_all = "snake_case")]
pub enum ImageField {
    #[serde(rename = "image")]
    Image,
    #[serde(rename = "image_url")]
    ImageUrl,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_image_field")]
    pub image_field: ImageField,
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

fn default_image_field() -> ImageField {
    ImageField::ImageUrl
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
    "structured_extraction"
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
