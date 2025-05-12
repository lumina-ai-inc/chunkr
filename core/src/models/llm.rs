use crate::configs::llm_config::Config;
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::Display;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, Display, PartialEq, Eq, Default)]
/// Specifies the fallback strategy for LLM processing
///
/// This can be:
/// 1. None - No fallback will be used
/// 2. Default - The system default fallback model will be used
/// 3. Model - A specific model ID will be used as fallback (check the documentation for the models.)
pub enum FallbackStrategy {
    /// No fallback will be used
    None,
    /// Use the system default fallback model
    #[default]
    Default,
    /// Use a specific model as fallback
    Model(String),
}

// Manual implementation of ToSql for FallbackStrategy
impl ToSql for FallbackStrategy {
    fn to_sql(
        &self,
        ty: &postgres_types::Type,
        out: &mut postgres_types::private::BytesMut,
    ) -> Result<postgres_types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        let s = match self {
            FallbackStrategy::None => "none".to_string(),
            FallbackStrategy::Default => "default".to_string(),
            FallbackStrategy::Model(id) => format!("model:{}", id),
        };
        s.to_sql(ty, out)
    }

    fn accepts(ty: &postgres_types::Type) -> bool {
        <String as ToSql>::accepts(ty)
    }

    postgres_types::to_sql_checked!();
}

// Manual implementation of FromSql for FallbackStrategy
impl<'a> FromSql<'a> for FallbackStrategy {
    fn from_sql(
        ty: &postgres_types::Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn std::error::Error + Sync + Send>> {
        let s = String::from_sql(ty, raw)?;
        match s.as_str() {
            "none" => Ok(FallbackStrategy::None),
            "default" => Ok(FallbackStrategy::Default),
            #[allow(clippy::manual_strip)]
            s if s.starts_with("model:") => Ok(FallbackStrategy::Model(s[6..].to_string())),
            _ => Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid FallbackStrategy format",
            ))),
        }
    }

    fn accepts(ty: &postgres_types::Type) -> bool {
        <String as FromSql>::accepts(ty)
    }
}

#[derive(Debug, Serialize, Clone, ToSql, FromSql, ToSchema)]
/// Controls the LLM used for the task.
pub struct LlmProcessing {
    /// The ID of the model to use for the task. If not provided, the default model will be used.
    /// Please check the documentation for the model you want to use.
    pub model_id: Option<String>,
    /// The fallback strategy to use for the LLMs in the task.
    #[serde(default)]
    pub fallback_strategy: FallbackStrategy,
    /// The maximum number of tokens to generate.
    pub max_completion_tokens: Option<u32>,
    /// The temperature to use for the LLM.
    #[serde(default)]
    pub temperature: f32,
}

impl<'de> Deserialize<'de> for LlmProcessing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Helper struct to deserialize the data initially
        #[derive(Deserialize)]
        struct LlmProcessingHelper {
            model_id: Option<String>,
            #[serde(default)]
            fallback_strategy: FallbackStrategy,
            max_completion_tokens: Option<u32>,
            #[serde(default)]
            temperature: f32,
        }

        let mut helper = LlmProcessingHelper::deserialize(deserializer)?;

        // Handle None or empty string case - get default model ID
        if helper.model_id.is_none()
            || helper
                .model_id
                .as_ref()
                .is_some_and(|id| id.trim().is_empty())
        {
            // Use the Config to get the default model ID
            if let Ok(config) = Config::from_env() {
                if let Ok(default_model) = config.get_model(None) {
                    helper.model_id = Some(default_model.id);
                }
            }
        }

        if helper.fallback_strategy == FallbackStrategy::Default {
            if let Ok(config) = Config::from_env() {
                if let Ok(Some(default_fallback_model)) =
                    config.get_fallback_model(FallbackStrategy::Default)
                {
                    helper.fallback_strategy = FallbackStrategy::Model(default_fallback_model.id);
                }
            }
        }

        // Return the processed struct
        Ok(LlmProcessing {
            model_id: helper.model_id,
            fallback_strategy: helper.fallback_strategy,
            max_completion_tokens: helper.max_completion_tokens,
            temperature: helper.temperature,
        })
    }
}

impl Default for LlmProcessing {
    fn default() -> Self {
        Self {
            model_id: None,
            fallback_strategy: FallbackStrategy::default(),
            max_completion_tokens: None,
            temperature: 0.0,
        }
    }
}
