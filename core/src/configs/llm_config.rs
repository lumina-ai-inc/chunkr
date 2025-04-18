use crate::models::llm::{FallbackStrategy, LlmProcessing};
use crate::models::open_ai::Message;
use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::RwLock;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    fallback_model: Option<String>,
    key: Option<String>,
    model: Option<String>,
    url: Option<String>,
    pub llm_models: Option<Vec<LlmModel>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmModel {
    pub id: String,
    pub model: String,
    pub provider_url: String,
    pub api_key: String,
    #[serde(default)]
    pub default: bool,
    #[serde(default)]
    pub fallback: bool,
    pub rate_limit: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlmModelPublic {
    pub id: String,
    pub default: bool,
    pub fallback: bool,
}

impl From<LlmModel> for LlmModelPublic {
    fn from(model: LlmModel) -> Self {
        LlmModelPublic {
            id: model.id,
            default: model.default,
            fallback: model.fallback,
        }
    }
}

static CONFIG: Lazy<RwLock<Option<Result<Config, ConfigError>>>> = Lazy::new(|| RwLock::new(None));

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        if let Some(config) = CONFIG.read().unwrap().as_ref() {
            return match config {
                Ok(cfg) => Ok(cfg.clone()),
                Err(e) => Err(ConfigError::Message(format!("{}", e))),
            };
        }

        let config = Self::load_config_from_env()?;

        *CONFIG.write().unwrap() = Some(Ok(config.clone()));

        Ok(config)
    }

    fn load_config_from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();
        let mut config = ConfigTrait::builder()
            .add_source(config::Environment::default().prefix("LLM").separator("__"))
            .build()?
            .try_deserialize::<Self>()?;

        if let Ok(models_path) = std::env::var("LLM__MODELS_PATH") {
            match Self::load_models_from_file(&models_path) {
                Ok(Some(models)) => {
                    config.llm_models = Some(models);
                }
                Ok(None) => {
                    println!("No models were found or loaded");
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        if config.llm_models.is_none()
            && config.model.is_some()
            && config.url.is_some()
            && config.key.is_some()
        {
            let default_model = LlmModel {
                id: "default".to_string(),
                model: config.model.clone().unwrap(),
                provider_url: config.url.clone().unwrap(),
                api_key: config.key.clone().unwrap(),
                default: true,
                fallback: false,
                rate_limit: None,
            };

            let fallback_model = LlmModel {
                id: "fallback".to_string(),
                model: config
                    .fallback_model
                    .clone()
                    .unwrap_or_else(|| config.model.clone().unwrap()),
                provider_url: config.url.clone().unwrap(),
                api_key: config.key.clone().unwrap(),
                default: false,
                fallback: true,
                rate_limit: None,
            };

            config.llm_models = Some(vec![default_model, fallback_model]);
            println!("Created default and fallback models from environment variables");
        }

        if config.llm_models.is_none() && config.model.is_none() {
            return Err(ConfigError::Message(
                "No models defined in config file or environment variables".to_string(),
            ));
        }

        Ok(config)
    }

    fn load_models_from_file(file_path: &str) -> Result<Option<Vec<LlmModel>>, ConfigError> {
        let contents = fs::read_to_string(file_path).map_err(|_| {
            ConfigError::Message(format!("Could not read file at path: {}", file_path))
        })?;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&contents)
            .map_err(|e| ConfigError::Message(format!("Error parsing YAML: {:?}", e)))?;

        let models = yaml
            .get("models")
            .ok_or_else(|| ConfigError::Message("No 'models' key found in YAML".to_string()))?;

        let parsed_models: Vec<LlmModel> = serde_yaml::from_value(models.clone())
            .map_err(|e| ConfigError::Message(format!("Error parsing models: {:?}", e)))?;

        Self::validate_models(&parsed_models)
            .map_err(|err| ConfigError::Message(format!("Invalid model configuration: {}", err)))?;

        println!("Successfully loaded models configuration");
        Ok(Some(parsed_models))
    }

    fn validate_models(models: &[LlmModel]) -> Result<(), String> {
        if models.is_empty() {
            return Err("No models defined".to_string());
        }

        let default_count = models.iter().filter(|m| m.default).count();
        let fallback_count = models.iter().filter(|m| m.fallback).count();

        if default_count != 1 {
            return Err(format!(
                "Exactly one model must be set as default, found {}",
                default_count
            ));
        }

        if fallback_count != 1 {
            return Err(format!(
                "Exactly one model must be set as fallback, found {}",
                fallback_count
            ));
        }

        Ok(())
    }

    fn get_model_by_id(&self, id: &str) -> Result<LlmModel, ConfigError> {
        self.llm_models
            .as_ref()
            .ok_or_else(|| ConfigError::Message("No LLM models configured".to_string()))?
            .iter()
            .find(|model| model.id == id)
            .cloned()
            .ok_or_else(|| ConfigError::Message("No model found".to_string()))
    }

    pub fn get_model(&self, id: Option<String>) -> Result<LlmModel, ConfigError> {
        if let Some(id) = id {
            self.get_model_by_id(&id)
        } else {
            self.llm_models
                .as_ref()
                .ok_or_else(|| ConfigError::Message("No LLM models configured".to_string()))?
                .iter()
                .find(|model| model.default)
                .cloned()
                .ok_or_else(|| ConfigError::Message("No default model found".to_string()))
        }
    }

    pub fn get_fallback_model(
        &self,
        fallback_strategy: FallbackStrategy,
    ) -> Result<Option<LlmModel>, ConfigError> {
        match fallback_strategy {
            FallbackStrategy::Default => {
                let default_fallback_model = self
                    .llm_models
                    .as_ref()
                    .ok_or_else(|| ConfigError::Message("No LLM models configured".to_string()))?
                    .iter()
                    .find(|model| model.fallback)
                    .cloned()
                    .ok_or_else(|| ConfigError::Message("No fallback model found".to_string()))?;

                Ok(Some(default_fallback_model))
            }
            FallbackStrategy::Model(model_id) => Ok(Some(self.get_model_by_id(&model_id)?)),
            FallbackStrategy::None => Ok(None),
        }
    }

    pub fn validate_llm_processing(&self, llm_processing: &LlmProcessing) -> Result<(), String> {
        // Validate primary model_id if specified
        if let Some(model_id) = &llm_processing.model_id {
            match self.get_model(Some(model_id.clone())) {
                Ok(_) => {}
                Err(_) => {
                    let available_models = self
                        .llm_models
                        .as_ref()
                        .map(|models| {
                            models
                                .iter()
                                .map(|m| m.id.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();

                    return Err(format!(
                        "Unknown model_id '{}'. Supported model IDs are: {}",
                        model_id, available_models
                    ));
                }
            }
        }

        // Validate fallback strategy if it's a specific model ID
        if let FallbackStrategy::Model(fallback_model_id) = &llm_processing.fallback_strategy {
            match self.get_model(Some(fallback_model_id.clone())) {
                Ok(_) => {}
                Err(_) => {
                    let available_models = self
                        .llm_models
                        .as_ref()
                        .map(|models| {
                            models
                                .iter()
                                .map(|m| m.id.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();

                    return Err(format!(
                        "Unknown fallback model_id '{}'. Supported model IDs are: {}",
                        fallback_model_id, available_models
                    ));
                }
            }
        }

        Ok(())
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
) -> Result<Vec<Message>, Box<dyn std::error::Error + Send + Sync>> {
    let template_json = load_prompt_template(template_name)?;
    let filled_json = substitute_template_placeholders(&template_json, values)?;
    let messages: Vec<Message> = serde_json::from_str(&filled_json)?;
    Ok(messages)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    async fn test_md_table_template() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut values = HashMap::new();
        values.insert(
            "image_url".to_string(),
            "https://example.com/image.jpg".to_string(),
        );
        let messages = create_messages_from_template("md_table", &values)?;
        println!("Message: {:?}", messages);
        Ok(())
    }

    #[tokio::test]
    async fn test_all_templates() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut values = HashMap::new();
        values.insert(
            "image_url".to_string(),
            "https://example.com/image.jpg".to_string(),
        );
        for &(template_name, _) in PROMPT_TEMPLATES {
            let messages = create_messages_from_template(template_name, &values)?;
            println!("Message: {:?}", messages);
        }
        Ok(())
    }
}
