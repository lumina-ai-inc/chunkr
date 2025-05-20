use opentelemetry::KeyValue;
use opentelemetry::{trace::TraceContextExt, Context};
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
    pub fn get_attributes(&self) -> Vec<KeyValue> {
        let mut attributes = vec![KeyValue::new("user.user_id", self.user_id.clone())];

        if self.api_key.is_some() {
            attributes.push(KeyValue::new("user.api_key_present", true));
        }

        if let Some(email) = &self.email {
            attributes.push(KeyValue::new("user.email", email.clone()));
        }

        if let Some(first_name) = &self.first_name {
            attributes.push(KeyValue::new("user.first_name", first_name.clone()));
        }

        if let Some(last_name) = &self.last_name {
            attributes.push(KeyValue::new("user.last_name", last_name.clone()));
        }

        attributes
    }

    pub fn add_attributes_to_ctx(&self) {
        let context = Context::current();
        let span = context.span();

        for attribute in self.get_attributes() {
            span.set_attribute(attribute);
        }
    }
}
