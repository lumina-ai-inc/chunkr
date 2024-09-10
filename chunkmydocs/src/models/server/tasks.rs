use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct TasksQuery {
    pub page: Option<i32>,
    pub limit: Option<i32>,
}



