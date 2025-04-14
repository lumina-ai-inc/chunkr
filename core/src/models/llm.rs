use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use strum_macros::Display;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema, Display)]
/// Specifies the fallback strategy for LLM processing
///
/// This can be:
/// 1. None - No fallback will be used
/// 2. Default - The system default fallback model will be used
/// 3. Custom - A specific model ID will be used as fallback (check the documentation for the models.)
pub enum FallbackStrategy {
    /// No fallback will be used
    None,
    /// Use the system default fallback model
    Default,
    /// Use a specific model as fallback
    String(String),
}

// Default implementation for FallbackStrategy
impl Default for FallbackStrategy {
    fn default() -> Self {
        FallbackStrategy::Default
    }
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
            FallbackStrategy::String(id) => format!("string:{}", id),
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
            s if s.starts_with("custom:") => Ok(FallbackStrategy::String(s[7..].to_string())),
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

#[derive(Debug, Serialize, Deserialize, Clone, ToSql, FromSql, ToSchema)]
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
