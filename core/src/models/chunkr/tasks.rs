use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct TasksQuery {
    pub page: Option<i64>,
    pub limit: Option<i64>,
}



