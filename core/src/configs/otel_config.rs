use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::Resource;
use opentelemetry_semantic_conventions::resource;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_service_namespace")]
    pub service_namespace: String,
    #[serde(default = "default_deployment_environment")]
    pub deployment_environment: String,
}

#[derive(Debug, Clone, Copy, Display, EnumString)]
pub enum ServiceName {
    #[strum(serialize = "chunkr-server")]
    Server,
    #[strum(serialize = "chunkr-task-worker")]
    TaskWorker,
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

    pub fn init_tracer(&self, service: ServiceName) -> Result<(), Box<dyn std::error::Error>> {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let otlp_exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(&self.endpoint)
            .build()?;

        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_batch_exporter(otlp_exporter)
            .with_resource(
                Resource::builder_empty()
                    .with_attributes([
                        KeyValue::new(resource::SERVICE_NAME, service.to_string()),
                        KeyValue::new(resource::SERVICE_NAMESPACE, self.service_namespace.clone()),
                        KeyValue::new(
                            "deployment.environment",
                            self.deployment_environment.clone(),
                        ),
                        KeyValue::new(resource::SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
                    ])
                    .build(),
            )
            .build();

        global::set_tracer_provider(provider);

        Ok(())
    }

    pub fn get_resource_attributes(&self, service: ServiceName) -> String {
        format!(
            "service.name={},service.namespace={},deployment.environment={}",
            service.to_string(),
            self.service_namespace,
            self.deployment_environment
        )
    }

    pub fn get_tracer(&self, service: ServiceName) -> opentelemetry::global::BoxedTracer {
        opentelemetry::global::tracer(service.to_string())
    }
}
