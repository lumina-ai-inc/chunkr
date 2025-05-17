use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_service_namespace")]
    pub service_namespace: String,
    #[serde(default = "default_deployment_environment")]
    pub deployment_environment: String,
}

fn default_endpoint() -> String {
    "http://localhost:4317".to_string()
}

fn default_service_namespace() -> String {
    "chunkr".to_string()
}

fn default_deployment_environment() -> String {
    "dev".to_string()
}

impl Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        dotenv_override().ok();

        // Try to load from OTEL_ environment variables first
        let mut builder = ConfigTrait::builder();

        // Add OpenTelemetry standard environment variables
        builder = builder.add_source(
            config::Environment::default()
                .prefix("OTEL_EXPORTER_OTLP")
                .separator("_"),
        );

        // Add our custom environment variables
        builder = builder.add_source(
            config::Environment::default()
                .prefix("OTEL")
                .separator("__"),
        );

        builder.build()?.try_deserialize()
    }

    pub fn get_resource_attributes(&self, service_name: &str) -> String {
        format!(
            "service.name={},service.namespace={},deployment.environment={}",
            service_name, self.service_namespace, self.deployment_environment
        )
    }
}
