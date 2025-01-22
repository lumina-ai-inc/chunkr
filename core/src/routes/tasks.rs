use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::task::TaskResponse;
use crate::models::chunkr::tasks::TasksQuery;
use crate::utils::routes::get_tasks::get_tasks;
use actix_web::{web, Error, HttpResponse};

/// Get Tasks
///
/// Retrieves a list of tasks
///
/// Example usage:
/// `GET /api/v1/tasks?page=1&limit=10&include_output=false`
#[utoipa::path(
    get,
    path = "/tasks",
    context_path = "/api/v1",
    tag = "Tasks",
    params(
        ("page" = Option<i64>, Query, description = "Page number"),
        ("limit" = Option<i64>, Query, description = "Number of tasks per page"),
        ("include_output" = Option<bool>, Query, description = "Whether to include task output in the response"),
    ),
    responses(
        (status = 200, description = "Detailed information describing the task", body = Vec<TaskResponse>),
        (status = 500, description = "Internal server error related to getting the task", body = String),
    ),
    security(
        ("api_key" = []),
    )
)]
pub async fn get_tasks_route(
    query: web::Query<TasksQuery>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let page = query.page.unwrap_or(1);
    let limit = query.limit.unwrap_or(10);
    let include_output = query.include_output.unwrap_or(false);
    let tasks = get_tasks(user_info.user_id.clone(), page, limit, include_output).await?;
    Ok(HttpResponse::Ok().json(tasks))
}
