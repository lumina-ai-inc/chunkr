use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::error::Error;
use std::fmt;
use uuid::Uuid;

/// Common context passed to all agents
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub user_id: String,
    pub entity_id: Option<String>,
    pub entity_type: Option<String>,
    pub metadata: Option<JsonValue>,
}

/// Standardized agent result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub success: bool,
    pub output: Option<JsonValue>,
    pub error: Option<String>,
    pub execution_id: String,
    pub tokens_used: Option<i32>,
    pub execution_time_ms: Option<i32>,
}

impl AgentResult {
    pub fn success(output: JsonValue, tokens_used: Option<i32>, execution_time_ms: i32) -> Self {
        Self {
            success: true,
            output: Some(output),
            error: None,
            execution_id: Uuid::new_v4().to_string(),
            tokens_used,
            execution_time_ms: Some(execution_time_ms),
        }
    }

    pub fn failure(error: String) -> Self {
        Self {
            success: false,
            output: None,
            error: Some(error),
            execution_id: Uuid::new_v4().to_string(),
            tokens_used: None,
            execution_time_ms: None,
        }
    }
}

/// Agent-specific errors
#[derive(Debug)]
pub enum AgentError {
    InputError(String),
    ProcessingError(String),
    ExternalServiceError(String),
    DatabaseError(String),
    ConfigurationError(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AgentError::InputError(msg) => write!(f, "Input Error: {}", msg),
            AgentError::ProcessingError(msg) => write!(f, "Processing Error: {}", msg),
            AgentError::ExternalServiceError(msg) => write!(f, "External Service Error: {}", msg),
            AgentError::DatabaseError(msg) => write!(f, "Database Error: {}", msg),
            AgentError::ConfigurationError(msg) => write!(f, "Configuration Error: {}", msg),
        }
    }
}

impl Error for AgentError {}

/// Log agent execution to database
pub async fn log_execution(
    agent_type: &str,
    entity_id: Option<String>,
    entity_type: Option<String>,
    input: JsonValue,
    output: Option<JsonValue>,
    status: &str,
    error: Option<String>,
    llm_provider: Option<String>,
    model: Option<String>,
    tokens_used: Option<i32>,
    execution_time_ms: Option<i32>,
) -> Result<String, Box<dyn Error + Send + Sync>> {
    use crate::utils::clients::get_pg_client;

    let execution_id = Uuid::new_v4().to_string();
    let now = Utc::now().naive_utc();
    let completed_at = if status == "completed" || status == "failed" {
        Some(now)
    } else {
        None
    };

    let client = get_pg_client().await?;

    client.execute(
        "INSERT INTO agent_executions (execution_id, agent_type, entity_id, entity_type, input, output, status, error, llm_provider, model, tokens_used, execution_time_ms, created_at, completed_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
        &[
            &execution_id,
            &agent_type,
            &entity_id,
            &entity_type,
            &input,
            &output,
            &status,
            &error,
            &llm_provider,
            &model,
            &tokens_used,
            &execution_time_ms,
            &now,
            &completed_at,
        ]
    ).await?;

    Ok(execution_id)
}

/// Convert agent error to user-friendly message
pub fn handle_agent_error(error: AgentError) -> String {
    match error {
        AgentError::InputError(msg) => {
            format!("Invalid input: {}", msg)
        }
        AgentError::ProcessingError(msg) => {
            format!("Processing failed: {}", msg)
        }
        AgentError::ExternalServiceError(msg) => {
            format!("External service error: {}. Please try again later.", msg)
        }
        AgentError::DatabaseError(_) => {
            "Database error occurred. Please try again.".to_string()
        }
        AgentError::ConfigurationError(_) => {
            "System configuration error. Please contact support.".to_string()
        }
    }
}

