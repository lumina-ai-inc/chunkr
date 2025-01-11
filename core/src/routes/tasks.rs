use crate::models::chunkr::auth::UserInfo;
use crate::models::chunkr::tasks::TasksQuery;
use crate::utils::routes::get_tasks::get_tasks;
use actix_web::{web, Error, HttpResponse};

pub async fn get_tasks_route(
    query: web::Query<TasksQuery>,
    user_info: web::ReqData<UserInfo>,
) -> Result<HttpResponse, Error> {
    let page = query.page.unwrap_or(1);
    let limit = query.limit.unwrap_or(10);

    let tasks = get_tasks(user_info.user_id.clone(), page, limit).await?;
    Ok(HttpResponse::Ok().json(tasks))
}
