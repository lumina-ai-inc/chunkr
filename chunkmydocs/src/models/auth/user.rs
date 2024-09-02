use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};
use chrono::{DateTime, Utc};

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Display, EnumString)]
pub enum Tier {
    Free,
    PayAsYouGo,
    Enterprise
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct UserResponse {
    pub user_id: String,
    pub customer_id: String,
    pub email: String,
    pub first_name: String,
    pub last_name: String,
    pub api_keys: Vec<String>,
    pub tier: Tier,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

