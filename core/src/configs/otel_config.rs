use config::{Config as ConfigTrait, ConfigError};
use dotenvy::dotenv_override;
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::{global, trace::TraceContextExt, Context, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{propagation::TraceContextPropagator, Resource};
use opentelemetry_semantic_conventions::resource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

#[derive(Debug, Clone, Copy, Display, EnumString)]
pub enum SpanName {
    #[strum(serialize = "auth")]
    Auth,
    #[strum(serialize = "create_task")]
    CreateTask,
    #[strum(serialize = "get_task")]
    GetTask,
    #[strum(serialize = "update_task")]
    UpdateTask,
    #[strum(serialize = "delete_task")]
    DeleteTask,
    #[strum(serialize = "cancel_task")]
    CancelTask,
    #[strum(serialize = "process_task")]
    ProcessTask,
    #[strum(serialize = "pipeline_init")]
    PipelineInit,
}

#[derive(Debug, Clone, Copy, Display, EnumString)]
pub enum EventName {
    #[strum(serialize = "task_skipped")]
    TaskSkipped,
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
            service,
            self.service_namespace,
            self.deployment_environment
        )
    }

    pub fn get_tracer(&self, service: ServiceName) -> opentelemetry::global::BoxedTracer {
        opentelemetry::global::tracer(service.to_string())
    }

    pub fn extract_context_for_propagation() -> Option<String> {
        let current_context = Context::current();
        if current_context.span().is_recording() {
            let propagator = TraceContextPropagator::new();
            let mut carrier = HashMap::new();

            propagator.inject_context(&current_context, &mut carrier);

            match serde_json::to_string(&carrier) {
                Ok(serialized) => Some(serialized),
                Err(_) => None,
            }
        } else {
            None
        }
    }

    pub fn inject_context(serialized_context: Option<String>) -> Context {
        if let Some(context_str) = serialized_context {
            match serde_json::from_str::<HashMap<String, String>>(&context_str) {
                Ok(carrier) => {
                    let propagator = TraceContextPropagator::new();
                    propagator.extract(&carrier)
                }
                Err(_) => Context::current(),
            }
        } else {
            Context::current()
        }
    }
}

/// Extract LLM error attributes from a JSON response string and add them to the OpenTelemetry span
pub fn extract_llm_error_attributes(response_text: &str) -> Vec<opentelemetry::KeyValue> {
    let mut attributes = Vec::new();

    // Try to parse the response as a generic JSON Value
    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(response_text) {
        // Add provider and model info if available - handle these first as they're common
        if let Some(provider) = json_value.get("provider") {
            if let Some(provider_str) = provider.as_str() {
                attributes.push(opentelemetry::KeyValue::new(
                    "response_provider",
                    provider_str.to_string(),
                ));
            }
        }

        if let Some(model) = json_value.get("model") {
            if let Some(model_str) = model.as_str() {
                attributes.push(opentelemetry::KeyValue::new(
                    "response_model",
                    model_str.to_string(),
                ));
            }
        }

        // Check for Google AI Studio error format
        if let Some(error) = json_value.get("error") {
            attributes.push(opentelemetry::KeyValue::new("error_response", true));

            // Extract provider info if available
            if let Some(metadata) = error.get("metadata") {
                if let Some(provider) = metadata.get("provider_name") {
                    if let Some(provider_str) = provider.as_str() {
                        attributes.push(opentelemetry::KeyValue::new(
                            "error_provider",
                            provider_str.to_string(),
                        ));
                    }
                }

                // Extract raw error details if available
                if let Some(raw) = metadata.get("raw") {
                    if let Some(raw_str) = raw.as_str() {
                        // Try to parse the raw error as JSON for more details
                        if let Ok(raw_json) = serde_json::from_str::<serde_json::Value>(raw_str) {
                            if let Some(inner_error) = raw_json.get("error") {
                                // Extract error code
                                if let Some(code) = inner_error.get("code") {
                                    if let Some(code_num) = code.as_i64() {
                                        attributes.push(opentelemetry::KeyValue::new(
                                            "error_code",
                                            code_num,
                                        ));
                                    }
                                }

                                // Extract error message
                                if let Some(message) = inner_error.get("message") {
                                    if let Some(message_str) = message.as_str() {
                                        attributes.push(opentelemetry::KeyValue::new(
                                            "error_message",
                                            message_str.to_string(),
                                        ));
                                    }
                                }

                                // Extract error status
                                if let Some(status) = inner_error.get("status") {
                                    if let Some(status_str) = status.as_str() {
                                        attributes.push(opentelemetry::KeyValue::new(
                                            "error_status",
                                            status_str.to_string(),
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Extract top-level error attributes
            if let Some(message) = error.get("message") {
                if let Some(message_str) = message.as_str() {
                    attributes.push(opentelemetry::KeyValue::new(
                        "error_message_top",
                        message_str.to_string(),
                    ));
                }
            }

            if let Some(code) = error.get("code") {
                if let Some(code_num) = code.as_i64() {
                    attributes.push(opentelemetry::KeyValue::new(
                        "error_code_top",
                        code_num,
                    ));
                }
            }
        }

        // Check for direct error format in choices (as seen in Google Gemini responses)
        if let Some(choices) = json_value.get("choices") {
            if let Some(choice_array) = choices.as_array() {
                if !choice_array.is_empty() {
                    attributes.push(opentelemetry::KeyValue::new("has_choices", true));

                    if let Some(choice_error) = choice_array[0].get("error") {
                        attributes.push(opentelemetry::KeyValue::new("error_in_choice", true));

                        if let Some(message) = choice_error.get("message") {
                            if let Some(message_str) = message.as_str() {
                                attributes.push(opentelemetry::KeyValue::new(
                                    "choice_error_message",
                                    message_str.to_string(),
                                ));
                            }
                        }

                        if let Some(code) = choice_error.get("code") {
                            if let Some(code_num) = code.as_i64() {
                                attributes.push(opentelemetry::KeyValue::new(
                                    "choice_error_code",
                                    code_num,
                                ));
                            } else if let Some(code_str) = code.as_str() {
                                // Handle string error codes
                                attributes.push(opentelemetry::KeyValue::new(
                                    "choice_error_code_str",
                                    code_str.to_string(),
                                ));
                            }
                        }
                    }

                    // Extract finish_reason which can indicate errors
                    if let Some(finish_reason) = choice_array[0].get("finish_reason") {
                        if let Some(reason_str) = finish_reason.as_str() {
                            attributes.push(opentelemetry::KeyValue::new(
                                "finish_reason",
                                reason_str.to_string(),
                            ));
                        }
                    }

                    // Also check for native_finish_reason
                    if let Some(native_finish_reason) = choice_array[0].get("native_finish_reason")
                    {
                        if let Some(reason_str) = native_finish_reason.as_str() {
                            attributes.push(opentelemetry::KeyValue::new(
                                "native_finish_reason",
                                reason_str.to_string(),
                            ));
                        }
                    }

                    // For Google Gemini models, check the message structure even if there's an error
                    if let Some(message) = choice_array[0].get("message") {
                        if let Some(message_obj) = message.as_object() {
                            attributes.push(opentelemetry::KeyValue::new("has_message", true));

                            if let Some(role) = message_obj.get("role") {
                                if let Some(role_str) = role.as_str() {
                                    attributes.push(opentelemetry::KeyValue::new(
                                        "message_role",
                                        role_str.to_string(),
                                    ));
                                }
                            }

                            // Note whether response has partial content
                            if let Some(content) = message_obj.get("content") {
                                if let Some(content_str) = content.as_str() {
                                    if !content_str.is_empty() {
                                        attributes.push(opentelemetry::KeyValue::new(
                                            "has_partial_content",
                                            true,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    attributes
}
