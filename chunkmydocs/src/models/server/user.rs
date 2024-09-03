use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use chrono::{DateTime, Utc};
use postgres_types::{FromSql, ToSql};

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Display, EnumString, FromSql, ToSql)]
#[postgres(name = "tier")]
pub enum Tier {
    Free,
    PayAsYouGo,
    Enterprise
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Display, EnumString, Hash)]
pub enum UsageType {
    Fast,
    HighQuality,
    Segment
}

impl UsageType {
    pub fn get_unit(&self) -> String {
        match self {
            UsageType::Fast => "Page".to_string(),
            UsageType::HighQuality => "Page".to_string(),
            UsageType::Segment => "Segment".to_string(),
        }
    }
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct User {
    pub user_id: String,
    pub customer_id: Option<String>,
    pub email: Option<String>,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub api_keys: Vec<String>,
    pub tier: Tier,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub usages: Vec<Usage>
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Usage {
    pub usage: i32,
    pub usage_limit: i32,
    pub usage_type: String,
    pub unit: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>
}