//! # LLM Configuration Module
//!
//! This module provides configuration management for Large Language Models (LLMs) used throughout the application.
//! It supports multiple model configurations with different roles and visibility levels.
//!
//! ## Model Types and Roles
//!
//! ### Default Models
//! - **`default: true`** - The primary model used for general LLM operations when no specific model is requested
//! - Must have exactly one model marked as default
//! - Used as fallback when no excel-specific default model is configured
//!
//! ### Fallback Models  
//! - **`fallback: true`** - The backup model used when the primary model fails
//! - Must have exactly one model marked as fallback
//! - Used as fallback when no excel-specific fallback model is configured
//!
//! ### Excel-Specific Models
//! - **`excel_default: true`** - The primary model specifically for Excel/spreadsheet processing
//! - Optional - if not configured, falls back to the regular default model
//! - At most one model can be marked as excel_default
//! - **`excel_fallback: true`** - The backup model specifically for Excel/spreadsheet processing
//! - Optional - if not configured, no fallback is used (FallbackStrategy::None)
//! - At most one model can be marked as excel_fallback
//!
//! ## Public vs Private Model Information
//!
//! ### Private (`LlmModel`)
//! Contains sensitive information including:
//! - API keys (`api_key`)
//! - Provider URLs (`provider_url`)
//! - Excel-specific flags (`excel_default`, `excel_fallback`)
//! - Rate limiting configuration (`rate_limit`)
//!
//! ### Public (`LlmModelPublic`)
//! Contains only non-sensitive information exposed to clients:
//! - Model ID (`id`)
//! - General role flags (`default`, `fallback`)
//! - **Note**: Excel-specific flags are intentionally excluded
//!
//! ## Configuration Sources
//!
//! ### Environment Variables
//! - `LLM__MODELS_PATH`: Path to YAML file containing model configurations
//! - Legacy single-model configuration via `LLM__MODEL`, `LLM__URL`, `LLM__KEY`, etc.
//!
//! ### YAML Configuration Example
//! ```yaml
//! models:
//!   - id: "gpt-4-turbo"
//!     model: "gpt-4-turbo"
//!     provider_url: "https://api.openai.com/v1"
//!     api_key: "sk-..."
//!     default: true
//!     fallback: false
//!     excel_default: false
//!     excel_fallback: false
//!
//!   - id: "gpt-3.5-turbo"
//!     model: "gpt-3.5-turbo"
//!     provider_url: "https://api.openai.com/v1"
//!     api_key: "sk-..."
//!     default: false
//!     fallback: true
//!     excel_default: false
//!     excel_fallback: false
//!
//!   - id: "excel-specialist"
//!     model: "gemini-pro-2.5"
//!     provider_url: "https://api.google.com/v1"
//!     api_key: "AIza..."
//!     default: false
//!     fallback: false
//!     excel_default: true
//!     excel_fallback: false
//!
//!   - id: "excel-backup"
//!     model: "gpt-4-turbo"
//!     provider_url: "https://api.openai.com/v1"
//!     api_key: "sk-..."
//!     default: false
//!     fallback: false
//!     excel_default: false
//!     excel_fallback: true
//! ```
//!
//! ## Usage Patterns
//!
//! ### General LLM Operations
//! ```rust
//! let config = Config::from_env()?;
//! let model = config.get_model(None)?; // Uses default model
//! let fallback = config.get_fallback_model(FallbackStrategy::Default)?; // Uses fallback model
//! ```
//!
//! ### Excel-Specific Operations
//! ```rust
//! let config = Config::from_env()?;
//! let model = config.get_excel_model(None)?; // Uses excel_default if configured, otherwise default
//! let fallback = config.get_excel_fallback_model(FallbackStrategy::Default)?; // Uses excel_fallback if configured, otherwise None
//! ```

use crate::models::llm::{FallbackStrategy, LlmProcessing, LlmProvider};
use crate::models::open_ai::Message;
use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::RwLock;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    fallback_model: Option<String>,
    key: Option<String>,
    model: Option<String>,
    url: Option<String>,
    pub llm_models: Option<Vec<LlmModel>>,
    pub excel_parsing_model: Option<String>,
    pub excel_parsing_fallback_model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LlmModel {
    pub id: String,
    pub model: String,
    pub provider_url: String,
    pub api_key: String,
    pub default: bool,
    pub fallback: bool,
    pub excel_default: bool,
    pub excel_fallback: bool,
    pub rate_limit: Option<f32>,
    pub provider: LlmProvider,
}

impl<'de> Deserialize<'de> for LlmModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct LlmModelHelper {
            id: String,
            model: String,
            provider_url: String,
            api_key: String,
            #[serde(default)]
            default: bool,
            #[serde(default)]
            fallback: bool,
            #[serde(default)]
            excel_default: bool,
            #[serde(default)]
            excel_fallback: bool,
            rate_limit: Option<f32>,
        }

        let helper = LlmModelHelper::deserialize(deserializer)?;
        let provider = LlmProvider::from_url(&helper.provider_url);

        Ok(LlmModel {
            id: helper.id,
            model: helper.model,
            provider_url: helper.provider_url,
            api_key: helper.api_key,
            default: helper.default,
            fallback: helper.fallback,
            excel_default: helper.excel_default,
            excel_fallback: helper.excel_fallback,
            rate_limit: helper.rate_limit,
            provider,
        })
    }
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
                Err(e) => Err(ConfigError::Message(format!("{e}"))),
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
            let default_model_id = config.model.clone().unwrap();
            let default_model = LlmModel {
                id: default_model_id.clone(),
                model: default_model_id.clone(),
                provider_url: config.url.clone().unwrap(),
                api_key: config.key.clone().unwrap(),
                default: true,
                fallback: false,
                excel_default: false,
                excel_fallback: false,
                rate_limit: None,
                provider: LlmProvider::from_url(
                    &config
                        .url
                        .clone()
                        .ok_or(ConfigError::Message("No URL provided".to_string()))?,
                ),
            };

            let fallback_model_id = config
                .fallback_model
                .clone()
                .unwrap_or_else(|| default_model_id.clone());
            let fallback_model = LlmModel {
                id: fallback_model_id.clone(),
                model: fallback_model_id.clone(),
                provider_url: config.url.clone().unwrap(),
                api_key: config.key.clone().unwrap(),
                default: false,
                fallback: true,
                excel_default: false,
                excel_fallback: false,
                rate_limit: None,
                provider: LlmProvider::from_url(
                    &config
                        .url
                        .clone()
                        .ok_or(ConfigError::Message("No URL provided".to_string()))?,
                ),
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
            ConfigError::Message(format!("Could not read file at path: {file_path}"))
        })?;

        let yaml = serde_yaml::from_str::<serde_yaml::Value>(&contents)
            .map_err(|e| ConfigError::Message(format!("Error parsing YAML: {e:?}")))?;

        let models = yaml
            .get("models")
            .ok_or_else(|| ConfigError::Message("No 'models' key found in YAML".to_string()))?;

        let parsed_models: Vec<LlmModel> = serde_yaml::from_value(models.clone())
            .map_err(|e| ConfigError::Message(format!("Error parsing models: {e:?}")))?;

        Self::validate_models(&parsed_models)
            .map_err(|err| ConfigError::Message(format!("Invalid model configuration: {err}")))?;

        println!("Successfully loaded models configuration");
        Ok(Some(parsed_models))
    }

    fn validate_models(models: &[LlmModel]) -> Result<(), String> {
        if models.is_empty() {
            return Err("No models defined".to_string());
        }

        let default_count = models.iter().filter(|m| m.default).count();
        let fallback_count = models.iter().filter(|m| m.fallback).count();
        let excel_default_count = models.iter().filter(|m| m.excel_default).count();
        let excel_fallback_count = models.iter().filter(|m| m.excel_fallback).count();

        if default_count != 1 {
            return Err(format!(
                "Exactly one model must be set as default, found {default_count}"
            ));
        }

        if fallback_count != 1 {
            return Err(format!(
                "Exactly one model must be set as fallback, found {fallback_count}"
            ));
        }

        if excel_default_count > 1 {
            return Err(format!(
                "At most one model can be set as excel_default, found {excel_default_count}"
            ));
        }

        if excel_fallback_count > 1 {
            return Err(format!(
                "At most one model can be set as excel_fallback, found {excel_fallback_count}"
            ));
        }

        let mut seen_ids = std::collections::HashSet::new();
        for model in models {
            if !seen_ids.insert(&model.id) {
                return Err(format!("Duplicate model ID found: {}", model.id));
            }

            // Validate API key requirements based on provider
            if model.api_key.is_empty() && !matches!(model.provider, LlmProvider::VertexAI) {
                return Err(format!(
                    "API key is required for model '{}' with provider '{}'",
                    model.id, model.provider
                ));
            }
        }

        Ok(())
    }

    pub fn get_model_by_id(&self, id: &str) -> Result<LlmModel, ConfigError> {
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

    pub fn get_excel_model(&self, id: Option<String>) -> Result<LlmModel, ConfigError> {
        if let Some(id) = id {
            self.get_model_by_id(&id)
        } else {
            // First try to find excel-specific default model
            if let Some(excel_model) = self
                .llm_models
                .as_ref()
                .ok_or_else(|| ConfigError::Message("No LLM models configured".to_string()))?
                .iter()
                .find(|model| model.excel_default)
                .cloned()
            {
                return Ok(excel_model);
            }

            // Fall back to regular default model
            self.llm_models
                .as_ref()
                .ok_or_else(|| ConfigError::Message("No LLM models configured".to_string()))?
                .iter()
                .find(|model| model.default)
                .cloned()
                .ok_or_else(|| ConfigError::Message("No default model found".to_string()))
        }
    }

    pub fn get_excel_fallback_model(
        &self,
        fallback_strategy: FallbackStrategy,
    ) -> Result<Option<LlmModel>, ConfigError> {
        match fallback_strategy {
            FallbackStrategy::Default => {
                // First try to find excel-specific fallback model
                if let Some(excel_fallback_model) = self
                    .llm_models
                    .as_ref()
                    .ok_or_else(|| ConfigError::Message("No LLM models configured".to_string()))?
                    .iter()
                    .find(|model| model.excel_fallback)
                    .cloned()
                {
                    return Ok(Some(excel_fallback_model));
                }

                // Fall back to regular fallback model
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
                        "Unknown model_id '{model_id}'. Supported model IDs are: {available_models}"
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
                        "Unknown fallback model_id '{fallback_model_id}'. Supported model IDs are: {available_models}"
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
    "formula_extended",
    "html_caption",
    "html_caption_extended",
    "html_footnote",
    "html_footnote_extended",
    "html_list_item",
    "html_list_item_extended",
    "html_page_footer",
    "html_page_footer_extended",
    "html_page_header",
    "html_page_header_extended",
    "html_page",
    "html_picture",
    "html_picture_extended",
    "html_section_header",
    "html_section_header_extended",
    "html_table",
    "html_table_extended",
    "html_text",
    "html_text_extended",
    "html_title",
    "html_title_extended",
    "llm_segment",
    "llm_segment_extended",
    "md_caption",
    "md_caption_extended",
    "md_footnote",
    "md_footnote_extended",
    "md_list_item",
    "md_list_item_extended",
    "md_page_footer",
    "md_page_footer_extended",
    "md_page_header",
    "md_page_header_extended",
    "md_page",
    "md_picture",
    "md_picture_extended",
    "md_section_header",
    "md_section_header_extended",
    "md_table",
    "md_table_extended",
    "md_text",
    "md_text_extended",
    "md_title",
    "md_title_extended",
    "identify_excel_elements",
];

fn load_prompt_template(prompt_name: &str) -> Result<String, std::io::Error> {
    PROMPT_TEMPLATES
        .iter()
        .find(|&&(name, _)| name == prompt_name)
        .map(|(_, content)| content.to_string())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Prompt '{prompt_name}' not found"),
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

        template = template.replace(&format!("{{{key}}}"), escaped_value);
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
        println!("Template JSON: {prompt}");
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
        println!("Filled template: {filled_json}");

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
        println!("Message: {messages:?}");
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
            println!("Message: {messages:?}");
        }
        Ok(())
    }
}
