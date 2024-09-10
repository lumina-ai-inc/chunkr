use crate::models::server::tasks::TasksQuery;
use crate::models::auth::auth::UserInfo;
use crate::utils::server::get_tasks::get_tasks;
use crate::utils::db::deadpool_postgres::Pool;
use actix_web::{web, Error, HttpResponse};
use aws_sdk_s3::Client as S3Client;

/// Get Extraction Tasks
///
/// Get a list of extraction tasks for the user.
#[utoipa::path(
    get,
    path = "/tasks",
    context_path = "/api",
    tag = "task",
    params(
        ("page" = Option<i32>, Query, description = "Page number for pagination"),
        ("limit" = Option<i32>, Query, description = "Number of items per page"),
    ),
    responses(
        (status = 200, description = "List of tasks", body = Vec<Task>),
        (status = 500, description = "Internal server error related to getting the extraction tasks", body = String),
    ),
    security(
        ("bearer_auth" = []),
        ("api_key" = [])
    )
)]
pub async fn get_tasks_status(
    pool: web::Data<Pool>,
    s3_client: web::Data<S3Client>,
    query: web::Query<TasksQuery>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let page = query.page.unwrap_or(1);
    let limit = query.limit.unwrap_or(10);

    let tasks = get_tasks(&pool, &s3_client, user_info.user_id.clone(), page, limit).await?;
    Ok(HttpResponse::Ok().json(tasks))
}