use chrono::{DateTime, Utc};
use postgres_types::{FromSql, ToSql};
use serde::{Deserialize, Serialize};
use super::extraction::Model;
use strum_macros::{EnumString, Display};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TaskResponse {
    pub task_id: String,
    pub status: Status,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<String>,
    pub expiration_time: Option<DateTime<Utc>>,
    pub message: String,
    pub file_url: Option<String>,
    pub task_url: Option<String>,
    pub model: Model,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSql, FromSql, PartialEq, Eq, EnumString, Display)]
pub enum Status {
    Starting,
    Processing,
    Succeeded,
    Failed,
    Canceled,
}

