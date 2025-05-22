use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayCount {
    pub day: NaiveDate,
    pub pages: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayStatusCount {
    pub day: NaiveDate,
    pub status: String,
    pub pages: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub email: Option<String>,
    pub total_pages: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSummary {
    pub total_pages: i32,
    pub total_tasks: i32,
    pub email: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDetails {
    pub task_id: String,
    pub user_id: String,
    pub email: Option<String>,
    pub name: String,
    pub page_count: i32,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedTaskDetails {
    pub tasks: Vec<TaskDetails>,
    pub page: u32,
    pub per_page: u32,
}
