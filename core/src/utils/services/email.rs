use crate::configs::email_config;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
// use std::fmt;
use std::sync::Arc;

// #[derive(Debug)]
// struct EmailError(String);

// impl fmt::Display for EmailError {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "{}", self.0)
//     }
// }

// impl Error for EmailError {}

#[derive(Debug)]
pub struct EmailService {
    client: Client,
    config: email_config::Config,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmailResponse {
    #[serde(flatten)]
    pub data: serde_json::Value,
}

impl EmailService {
    pub fn new(config: email_config::Config) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub async fn send_welcome_email(
        &self,
        name: &str,
        email: &str,
    ) -> Result<EmailResponse, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/email/welcome", self.config.server_url);
        let response = self
            .client
            .post(&url)
            .query(&[("name", name), ("email", email)])
            .send()
            .await?
            .json()
            .await?;

        Ok(response)
    }

    pub async fn send_upgrade_email(
        &self,
        name: &str,
        email: &str,
        tier: &str,
    ) -> Result<EmailResponse, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/email/upgrade", self.config.server_url);
        let response = self
            .client
            .post(&url)
            .query(&[("name", name), ("email", email), ("tier", tier)])
            .send()
            .await?
            .json()
            .await?;

        Ok(response)
    }

    pub async fn send_reactivation_email(
        &self,
        name: &str,
        email: &str,
        cal_url: &str,
    ) -> Result<EmailResponse, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/email/reactivate", self.config.server_url);
        let response = self
            .client
            .post(&url)
            .query(&[("name", name), ("email", email), ("cal_url", cal_url)])
            .send()
            .await?
            .json()
            .await?;

        Ok(response)
    }

    pub async fn send_free_pages_email(
        &self,
        name: &str,
        email: &str,
    ) -> Result<EmailResponse, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/email/free-pages", self.config.server_url);
        let response = self
            .client
            .post(&url)
            .query(&[("name", name), ("email", email)])
            .send()
            .await?
            .json()
            .await?;

        Ok(response)
    }

    pub async fn send_unpaid_invoice_email(
        &self,
        name: &str,
        email: &str,
    ) -> Result<EmailResponse, Box<dyn Error + Send + Sync>> {
        let url = format!("{}/email/unpaid_invoice", self.config.server_url);
        let response = self
            .client
            .post(&url)
            .query(&[("name", name), ("email", email)])
            .send()
            .await?
            .json()
            .await?;

        Ok(response)
    }
}

pub type SharedEmailService = Arc<EmailService>;
