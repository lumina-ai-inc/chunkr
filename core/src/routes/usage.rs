use crate::models::chunkr::auth::UserInfo;
use crate::utils::clients::get_pg_client;
use actix_web::{web, Error, HttpResponse};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Serialize, Deserialize, ToSchema)]
struct UsageResponse {
    email: String,
    key: String,
    total_usage: i32,
    usage_limit: i32,
    usage_percentage: f64,
}

#[derive(Serialize, Deserialize, ToSchema)]
struct TaskCountResponse {
    task_count: i64,
}

/// Get Task Count
///
/// Retrieve the total number of tasks for the authenticated user
#[utoipa::path(
    get,
    path = "/usage/task_count",
    context_path = "/api",
    tag = "usage",
    responses(
        (status = 200, description = "Successfully retrieved task count", body = TaskCountResponse),
        (status = 500, description = "Internal server error", body = String),
    ),
    security(
        ("api_key" = [])
    )
)]
pub async fn get_task_count(api_info: web::ReqData<UserInfo>) -> Result<HttpResponse, Error> {
    let user_id = api_info.user_id.clone();

    let client = get_pg_client().await.map_err(|e| {
        eprintln!("Error connecting to database: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database connection error")
    })?;

    let stmt = client
        .prepare("SELECT COUNT(*) FROM tasks WHERE user_id = $1")
        .await
        .map_err(|e| {
            eprintln!("Error preparing statement: {:?}", e);
            actix_web::error::ErrorInternalServerError("Database query error")
        })?;

    let row = client.query_one(&stmt, &[&user_id]).await.map_err(|e| {
        eprintln!("Error executing query: {:?}", e);
        actix_web::error::ErrorInternalServerError("Database query error")
    })?;

    let task_count: i64 = row.get(0);

    Ok(HttpResponse::Ok().json(TaskCountResponse { task_count }))
}
