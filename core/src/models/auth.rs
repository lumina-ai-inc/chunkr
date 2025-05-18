use opentelemetry::KeyValue;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct UserInfo {
    pub user_id: String,
    pub api_key: Option<String>,
    pub email: Option<String>,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
}

impl UserInfo {
    pub fn add_trace_attributes<S>(&self, span: &mut S)
    where
        S: opentelemetry::trace::Span,
    {
        span.set_attribute(KeyValue::new("user.user_id", self.user_id.clone()));

        if let Some(_) = &self.api_key {
            span.set_attribute(KeyValue::new("user.api_key_present", true));
        }

        if let Some(email) = &self.email {
            span.set_attribute(KeyValue::new("user.email", email.clone()));
        }

        if let Some(first_name) = &self.first_name {
            span.set_attribute(KeyValue::new("user.first_name", first_name.clone()));
        }

        if let Some(last_name) = &self.last_name {
            span.set_attribute(KeyValue::new("user.last_name", last_name.clone()));
        }
    }
}
