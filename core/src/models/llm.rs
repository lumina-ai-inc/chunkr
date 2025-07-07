use crate::configs::llm_config::Config;
use postgres_types::{FromSql, ToSql};
use schemars::{schema_for, JsonSchema as SchemarsJsonSchema};
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
            FallbackStrategy::Model(id) => format!("model:{id}"),
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

impl LlmProcessing {
    /// Resolves default values from config when needed
    fn resolve_defaults(
        model_id: Option<String>,
        fallback_strategy: FallbackStrategy,
    ) -> (Option<String>, FallbackStrategy) {
        let mut resolved_model_id = model_id;
        let mut resolved_fallback_strategy = fallback_strategy;

        // Handle None or empty string case - get default model ID
        if resolved_model_id.is_none()
            || resolved_model_id
                .as_ref()
                .is_some_and(|id| id.trim().is_empty())
        {
            // Use the Config to get the default model ID
            if let Ok(config) = Config::from_env() {
                if let Ok(default_model) = config.get_model(None) {
                    resolved_model_id = Some(default_model.id);
                }
            }
        }

        // Resolve fallback strategy if it's Default
        if resolved_fallback_strategy == FallbackStrategy::Default {
            if let Ok(config) = Config::from_env() {
                if let Ok(Some(default_fallback_model)) =
                    config.get_fallback_model(FallbackStrategy::Default)
                {
                    resolved_fallback_strategy = FallbackStrategy::Model(default_fallback_model.id);
                }
            }
        }

        (resolved_model_id, resolved_fallback_strategy)
    }

    /// Create a new LLM processing struct
    ///
    /// This function will resolve the default values from the config if needed.
    ///
    /// ### Arguments
    ///
    /// * `model_id` - The ID of the model to use for the task. If not provided, the default model will be used.
    /// * `fallback_strategy` - The fallback strategy to use for the LLMs in the task. If not provided, the fallback strategy will be resolved to None.
    /// * `max_completion_tokens` - The maximum number of tokens to generate. If not provided, the default value will be used.
    /// * `temperature` - The temperature to use for the LLM. If not provided, the default value will be used.
    ///
    /// ### Returns
    ///
    /// A new `LlmProcessing` struct with the resolved default values.
    pub fn new(
        model_id: Option<String>,
        fallback_strategy: Option<FallbackStrategy>,
        max_completion_tokens: Option<u32>,
        temperature: f32,
    ) -> Self {
        let (resolved_model_id, resolved_fallback_strategy) = Self::resolve_defaults(
            model_id,
            fallback_strategy.unwrap_or(FallbackStrategy::None),
        );

        Self {
            model_id: resolved_model_id,
            fallback_strategy: resolved_fallback_strategy,
            max_completion_tokens,
            temperature,
        }
    }
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

        let helper = LlmProcessingHelper::deserialize(deserializer)?;

        // Use the resolve_defaults method to handle config-based defaults
        let (resolved_model_id, resolved_fallback_strategy) =
            Self::resolve_defaults(helper.model_id, helper.fallback_strategy);

        // Return the processed struct
        Ok(LlmProcessing {
            model_id: resolved_model_id,
            fallback_strategy: resolved_fallback_strategy,
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

#[derive(Debug, Clone, Serialize)]
pub struct JsonSchemaDefinition {
    pub name: String,
    pub description: Option<String>,
    pub schema: serde_json::Value,
}

impl JsonSchemaDefinition {
    pub fn new(name: String, description: Option<String>, schema: serde_json::Value) -> Self {
        Self {
            name,
            description,
            schema,
        }
    }

    pub fn from_struct<T: SchemarsJsonSchema>(name: String, description: Option<String>) -> Self {
        let schema = Self::generate_raw_schema_for_struct::<T>();
        Self::new(name, description, schema)
    }

    /// Generates a raw JSON schema from a Rust struct
    ///
    /// The struct must implement `SchemarsJsonSchema` and have doc comments on fields for descriptions.
    ///
    /// # Example
    /// ```rust
    /// use schemars::JsonSchema;
    /// use serde::{Deserialize, Serialize};
    ///
    /// #[derive(Serialize, Deserialize, JsonSchema)]
    /// struct MyStruct {
    ///     /// The name field
    ///     name: String,
    ///     /// The age field
    ///     age: u32,
    /// }
    ///
    /// let schema = generate_raw_schema_for_struct::<MyStruct>();
    /// ```
    pub fn generate_raw_schema_for_struct<T: SchemarsJsonSchema>() -> serde_json::Value {
        let schema = schema_for!(T);
        serde_json::to_value(schema).expect("Failed to serialize schema")
    }
}

#[derive(Debug, Clone, Display, Deserialize)]
pub enum LlmProvider {
    #[strum(serialize = "openai")]
    OpenAI,
    #[strum(serialize = "genai")]
    Genai,
    #[strum(serialize = "vertexai")]
    VertexAI,
}

impl LlmProvider {
    pub fn from_url(url: &str) -> Self {
        if url.contains("generativelanguage.googleapis.com") {
            Self::Genai
        } else if url.contains("aiplatform.googleapis.com") {
            Self::VertexAI
        } else {
            Self::OpenAI
        }
    }
}
