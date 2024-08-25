use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub enum AccessLevel {
    OWNER,
    ADMIN,
    WRITE,
    READ,
}

impl fmt::Display for AccessLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccessLevel::OWNER => write!(f, "OWNER"),
            AccessLevel::ADMIN => write!(f, "ADMIN"),
            AccessLevel::WRITE => write!(f, "WRITE"),
            AccessLevel::READ => write!(f, "READ"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub enum UsageType {
    FREE,
    PAID,
}

impl fmt::Display for UsageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UsageType::FREE => write!(f, "FREE"),
            UsageType::PAID => write!(f, "PAID"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ServiceType {
    EXTRACTION,
    SEARCH,
}

impl fmt::Display for ServiceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServiceType::EXTRACTION => write!(f, "EXTRACTION"),
            ServiceType::SEARCH => write!(f, "SEARCH"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiRequest {
    pub user_id: String,
    pub email: String,
    pub access_level: AccessLevel,
    pub expires_at: Option<DateTime<Utc>>,
    pub initial_usage: Option<i32>,
    pub usage_limit: Option<i32>,
    pub usage_type: Option<UsageType>,
    pub service_type: Option<ServiceType>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiKey {
    pub key: String,
    pub user_id: Option<String>,
    pub dataset_id: Option<String>,
    pub org_id: Option<String>,
    pub access_level: Option<AccessLevel>,
    pub active: Option<bool>,
    pub deleted: Option<bool>,
    pub created_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub deleted_at: Option<DateTime<Utc>>,
    pub deleted_by: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiKeyUsage {
    pub id: Option<i32>,
    pub api_key: String,
    pub usage: Option<i32>,
    pub usage_type: Option<UsageType>,
    pub created_at: Option<DateTime<Utc>>,
    pub service_type: Option<ServiceType>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApiKeyLimit {
    pub id: Option<i32>,
    pub api_key: String,
    pub usage_limit: Option<i32>,
    pub usage_type: Option<UsageType>,
    pub created_at: Option<DateTime<Utc>>,
    pub service_type: Option<ServiceType>,
}
